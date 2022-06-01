import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

def partition_dataset_df(
    df: pd.DataFrame,
    split: tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a Pandas dataframe into train, validation, and test subsets.

    Args:
        df (pd.DataFrame): Source dataframe.
        split (tuple[float, float, float], optional): Tuple of split ratios. Must sum to 1. Defaults to (0.8, 0.1, 0.1).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Tuple of train, validation, and test dataframes.
    """
    assert isinstance(split, (list, tuple))
    assert len(split) == 3
    assert np.isclose(sum(split), 1.0)

    # Split dataframe into train/val/test dataframes.
    # Inspiration: https://www.tensorflow.org/tutorials/structured_data/time_series#split_the_data
    train_split, val_split, test_split = split # Unpack split tuple.
    n = len(df.index) # Total number of data records.
    df_train = df[:int(n*train_split)].copy()
    df_val = df[int(n*train_split):int(n*(1-test_split))].copy()
    df_test = df[int(n*(1-test_split)):].copy()
    return df_train, df_val, df_test



# Inspiration: https://www.tensorflow.org/tutorials/structured_data/time_series#data_windowing
class WindowGenerator:
    def __init__(self,
        in_seq_len: int,
        out_seq_len: int,
        shift: int,
        columns: list[str],
        in_feat: list[str] = None,
        out_feat: list[str] = None,
        batch_size: int = 32,
        shuffle: bool = False,
        ):
        """Constructs sliding windows of sequential data.

        Args:
            in_seq_len (int): Input sequence length.
            out_seq_len (int): Output (target) sequence length.
            shift (int): Number of indices to skip between elements when traversing window.
            columns (list[str]): List of column names in dataframes.
            in_feat (list[str], optional): Desired subset of input features for window. Defaults to None.
            out_feat (list[str], optional): Desired subset of output features for window. Defaults to None.
            batch_size (int, optional): Batch size. Defaults to 32.
            shuffle (bool, optional): Shuffle windows prior to batching. Defaults to False.
        """
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Preserve sequence information.
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.shift = shift
        self.total_window_len = in_seq_len + shift

        # Setup indexing slices for window extraction.
        self.in_slice = slice(0, self.in_seq_len)
        self.out_slice = slice(self.total_window_len - self.out_seq_len, None)
        self.in_idx = np.arange(self.total_window_len)[self.in_slice]
        self.out_idx = np.arange(self.total_window_len)[self.out_slice]

        # Setup train/val/test column extractors.
        self.col2idx = {name: i for i, name in enumerate(columns)}
        if in_feat is not None:
            self.in_feat = in_feat
            self.in_col_idx = [self.col2idx[col] for col in in_feat]
        else:
            self.in_feat = [columns[i] for i in self.in_col_idx]
            self.in_col_idx = list(range(len(columns)))
        if out_feat is not None:
            self.out_feat = out_feat
            self.out_col_idx = [self.col2idx[col] for col in out_feat]
        else:
            self.out_feat = [columns[i] for i in self.out_col_idx]
            self.out_col_idx = list(range(len(columns)))

    def __repr__(self):
        """String representation of class."""
        return '\n'.join([
            f"Total window length: {self.total_window_len}",
            f"Input indices: {self.in_idx}",
            f"Output indices: {self.out_idx}",
            f"Input features: {self.in_feat}",
            f"Output features: {self.out_feat}",
        ])

    def split_window(self, 
        window: tf.Tensor, # window shape is (batch, seq, feat)
        ) -> tuple[tf.Tensor, tf.Tensor]:
        """Splits a single window of data into input/output seqments.

        Args:
            window (tf.Tensor): Tensor of window data with shape (batch, seq, feat).

        Returns:
            tuple[tf.Tensor, tf.Tensor]: 2-tuple of input/output data segments, where the shapes are:
                - Input window: (batch, in_seq_len, in_feat)
                - Output window: (batch, out_seq_len, out_feat)
        """
        # Decompose input/output sequence from given input window.
        in_seq = tf.stack([window[:, self.in_slice, i] for i in self.in_col_idx], axis=-1)
        out_seq = tf.stack([window[:, self.out_slice, i] for i in self.out_col_idx], axis=-1)

        # Set shape for input/output sequences.
        # Note that dimensions set to `None` are not updated.
        in_seq = tf.ensure_shape(in_seq, (None, self.in_seq_len, None))
        out_seq = tf.ensure_shape(out_seq, (None, self.out_seq_len, None))

        return in_seq, out_seq

    def make_dataset(self, 
        df: pd.DataFrame,
        batch_size: int = None,
        shuffle: bool = None,
        ) -> tf.data.Dataset:
        """Construct a TensorFlow Dataset from given input data frame.

        Datasets load tuples of batched input/output windows with shapes:
            - Input window: (batch, in_seq_len, in_feat)
            - Output window: (batch, out_seq_len, out_feat)

        Note that output windows are generally target sequences.

        Args:
            df (pd.DataFrame): Source data frame.
            batch_size (int, optional): Batch size. Defaults to class value.
            shuffle (bool, optional): Shuffle windows prior to batching. Defaults to class value.

        Returns:
            tf.data.Dataset: Dataset object.
        """
        if batch_size is None:
            batch_size = self.batch_size
        if shuffle is None:
            shuffle = self.shuffle

        # Convert data frame into numpy matrix.
        data = df.to_numpy()

        # Convert data matrix into TensorFlow dataset.
        # dataset = keras.utils.timeseries_dataset_from_array(
        dataset = keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_len,
            sequence_stride=self.shift,
            shuffle=shuffle,
            batch_size=batch_size,
        )

        # Pipe the raw dataset into the window splitting function.
        dataset = dataset.map(self.split_window)

        # Return the dataset.
        return dataset