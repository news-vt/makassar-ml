from __future__ import annotations
import os
import pandas as pd
from pathlib import Path
import tensorflow as tf
import tensorflow.keras as keras
from .utility import partition_dataset_df, WindowGenerator

def load_beijingpm25_df(
    path: str = None,
    split: tuple[float, float, float] = None,
    ) -> pd.DataFrame:
    """Loads Beijing PM2.5 dataset as a pandas dataframe.

    https://archive-beta.ics.uci.edu/ml/datasets/beijing+pm2+5+data

    Dataset features:
        - `No`: (NOT USED) row number
        - `year`: year of data in this row
        - `month`: month of data in this row
        - `day`: day of data in this row
        - `hour`: hour of data in this row
        - `pm2.5`: PM2.5 concentration (ug/m^3)
        - `DEWP`: Dew Point (â„ƒ)
        - `TEMP`: Temperature (â„ƒ)
        - `PRES`: Pressure (hPa)
        - `cbwd`: (NOT USED) Combined wind direction
        - `Iws`: Cumulated wind speed (m/s)
        - `Is`: Cumulated hours of snow
        - `Ir`: Cumulated hours of rain
        - `datetime`: (NOT USED) dynamically generated datetime string
    """
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv'

    # Set path to current directory if necessary.
    if path is None:
        path = os.getcwd()

    # Convert path to `Path` object.
    if not isinstance(path, Path):
        path = Path(path)

    # Create path if necessary.
    path.mkdir(parents=True, exist_ok=True)

    # Download the dataset if necessary.
    filename = Path(url.rsplit('/', 1)[1]) # Get name of file.
    filepath = (path / filename).expanduser().resolve()
    filepath = keras.utils.get_file(
        fname=filepath,
        origin=url,
        )

    # Load as Pandas DataFrame.
    df = pd.read_csv(filepath)

    # Create single date column from independent year/month/day columns.
    df['datetime'] = pd.to_datetime(df[['year','month','day','hour']])

    # Add day of year column.
    df['day_of_year'] = df['datetime'].dt.day_of_year

    # Partition dataframe into train/val/test.
    if split is not None:
        train_df, val_df, test_df = partition_dataset_df(df, split=split)
        return train_df, val_df, test_df

    # Use entire dataset.
    else:
        return df


def load_beijingpm25_ds(
    in_seq_len: int,
    out_seq_len: int,
    shift: int,
    in_feat: list[str],
    out_feat: list[str],
    path: str = None,
    split: tuple[float, float, float] = None,
    batch_size: int = 32,
    shuffle: bool = False,
    # drop_columns: list[str] = ['No', 'year', 'month', 'day', 'hour', 'datetime', 'cbwd'],
    drop_columns: list[str] = ['datetime', 'cbwd'],
    drop_nan: bool = True,
    norm: bool = True,
    ) -> tf.data.Dataset|tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:

    # Load dataframe.
    df = load_beijingpm25_df(path=path)
    if drop_columns:
        df.drop(columns=drop_columns, inplace=True)
    if drop_nan:
        df.dropna(inplace=True) # Remove NaN.

    # Build window generator.
    columns = df.columns
    wg = WindowGenerator(
        in_seq_len=in_seq_len,
        out_seq_len=out_seq_len,
        shift=shift,
        columns=columns,
        in_feat=in_feat,
        out_feat=out_feat,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    # Partition dataframe into train/val/test.
    if split is not None:
        train_df, val_df, test_df = partition_dataset_df(df, split=split)

        # Reset the indices to zero.
        train_df.reset_index(inplace=True)
        val_df.reset_index(inplace=True)
        test_df.reset_index(inplace=True)

        # Normalize to 0 mean and 1 std.
        if norm:
            train_mean = train_df.mean()
            train_std = train_df.std()
            train_df = (train_df - train_mean)/train_std
            val_df = (val_df - train_mean)/train_std
            test_df = (test_df - train_mean)/train_std

        # Convert train/val/test frames into windowed datasets.
        train_ds = wg.make_dataset(train_df)
        val_ds = wg.make_dataset(val_df)
        test_ds = wg.make_dataset(test_df)

        return train_ds, val_ds, test_ds

    # Use entire dataset.
    else:

        # Normalize to 0 mean and 1 std.
        if norm:
            mean = df.mean()
            std = df.std()
            df = (df - mean)/std

        # Build dataset.
        ds = wg.make_dataset(df)

        return ds


def load_data(*args, **kwargs):
    """Wrapper for loading Beijing PM2.5 dataset."""
    return load_beijingpm25_ds(*args, **kwargs)




# df = load_beijingpm25_df(
#     path=DATASET_ROOT/'beijing_pm25',
# )
# print(df.info())

# # a, b, c = partition_dataset_df(df)
# # print(a.info())
# # print(b.info())
# # print(c.info())


# in_seq_len = 30
# out_seq_len = 7
# shift = 1
# in_feat = [
#     # 'pm2.5',
#     'DEWP',
#     'TEMP',
#     'PRES',
#     'Iws',
#     'Is',
#     'Ir',
# ]
# out_feat = [
#     'pm2.5',
# ]
# train_ds, val_ds, test_ds = load_beijingpm25_ds(
#     in_seq_len=in_seq_len,
#     out_seq_len=out_seq_len,
#     shift=shift,
#     in_feat=in_feat,
#     out_feat=out_feat,
#     split=(0.8,0.1,0.1),
#     path=DATASET_ROOT/'beijing_pm25',
# )
# print(train_ds)

# logger.info(f"train: {tf.data.experimental.cardinality(train_ds)} batches")
# logger.info(f"val: {tf.data.experimental.cardinality(val_ds)} batches")
# logger.info(f"test: {tf.data.experimental.cardinality(test_ds)} batches")