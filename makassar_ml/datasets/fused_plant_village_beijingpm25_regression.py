from __future__ import annotations
import datetime as dt
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from .beijingpm25 import load_beijingpm25_df
from .plant_village import image_augmentation


def fuse_dated_images_timeseries(
    ds_images_dates: tf.data.Dataset,
    df_timeseries: pd.DataFrame,
    in_seq_len: int,
    out_seq_len: int,
    datetime_column: str = 'datetime',
    datetime_format: str = '%Y-%m-%d %H:%M:%S',
    in_features: list[str] = None,
    out_features: list[str] = None,
    ):

    # Set the index to datetime and integer so lookup is quicker.
    df_timeseries['integer_index'] = range(len(df_timeseries.index))
    df_timeseries_datetimeindex = df_timeseries.set_index([datetime_column,'integer_index'])

    def gen():
        for (image, date), _ in ds_images_dates:

            # Get history and image.
            date_obj = dt.datetime.strptime(date.numpy().decode('utf8'), datetime_format)
            date_idx = df_timeseries_datetimeindex.loc[date_obj].index.values[0]
            idx_start = date_idx - in_seq_len
            df_timeseries_range_in = df_timeseries_datetimeindex[idx_start:date_idx]
            df_timeseries_range_in.reset_index(inplace=True)
            if in_features is not None:
                df_timeseries_range_in = df_timeseries_range_in[in_features]
            tensor_timeseries_range_in = tf.convert_to_tensor(df_timeseries_range_in)

            # Get output values.
            idx_end = date_idx + out_seq_len
            df_timeseries_range_out = df_timeseries_datetimeindex[date_idx:idx_end]
            df_timeseries_range_out.reset_index(inplace=True)
            if out_features is not None:
                df_timeseries_range_out = df_timeseries_range_out[out_features]
            tensor_timeseries_range_out = tf.convert_to_tensor(df_timeseries_range_out)

            yield ((image, tensor_timeseries_range_in), tensor_timeseries_range_out)

    ds_fused = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            (
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8), # image
                tf.TensorSpec(shape=(None, None), dtype=tf.float32), # input timeseries
            ),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32), # output timeseries
        ),
    )
    # Set length of dataset since it will be the same as the image dataset.
    ds_fused = ds_fused.apply(tf.data.experimental.assert_cardinality(tf.data.experimental.cardinality(ds_images_dates)))
    return ds_fused


def load_data(
    timeseries_path: str,
    timeseries_reserve_offset_index_in: int,
    timeseries_reserve_offset_index_out: int,
    timeseries_datetime_column: str,
    timeseries_features_in: list[str],
    timeseries_features_out: list[str],
    timeseries_seq_len_in: dict,
    timeseries_seq_len_out: dict,
    image_shape: list[int, int, int],
    split: list[str, str, str],
    shuffle_files: bool,
    batch_size: int,
    with_info: bool = False,
    norm: bool = True,
    ) -> tuple[tf.data.Dataset,tf.data.Dataset,tf.data.Dataset]:

    assert len(split) == 3
    assert all(isinstance(item, str) for item in split)

    # Load image dataset.
    # ds_images_train, ds_images_val, ds_images_test = tfds.load(
    ds_images_tuple, info = tfds.load(
        name='plant_village',
        split=split,
        shuffle_files=shuffle_files,
        as_supervised=True,
        with_info=True,
    )

    # Load weather dataset.
    df_timeseries = load_beijingpm25_df(
        path=timeseries_path,
    )
    df_timeseries.dropna(inplace=True) # Remove NaN.

    # Normalize the time-series data.
    if norm:
        normed_featured = list(set(timeseries_features_in + timeseries_features_out))
        mean = df_timeseries[normed_featured].mean()
        std = df_timeseries[normed_featured].std()
        df_timeseries[normed_featured] = (df_timeseries[normed_featured] - mean)/std

    # Fuse weather data for each split.
    ds_fused_out = []
    for i in range(len(ds_images_tuple)):

        # Randomly select dates to associate with each element.
        n_images = int(ds_images_tuple[i].cardinality())
        random_dates = df_timeseries.iloc[timeseries_reserve_offset_index_in:-timeseries_reserve_offset_index_out].sample(n=n_images, replace=True)[timeseries_datetime_column]

        # Fuse images with dates.
        ds_images_dates = tf.data.Dataset.zip(
            (ds_images_tuple[i], tf.data.Dataset.from_tensor_slices(random_dates.astype(str)))
        ).map(
            lambda image_label, datestring: ((image_label[0], datestring), image_label[1]) # Reorder to ((image, date), label).
        )

        # Fuse the datasets.
        # Returns ((image, timeseries_in), timeseries_out)
        ds_fused = fuse_dated_images_timeseries(
            ds_images_dates=ds_images_dates, 
            df_timeseries=df_timeseries,
            in_seq_len=timeseries_seq_len_in,
            out_seq_len=timeseries_seq_len_out,
            in_features=timeseries_features_in,
            out_features=timeseries_features_out,
        )

        # Augment the images.
        # Flip the train/val.
        if i < 2:
            ds_fused = ds_fused.map(
                lambda image_data, data_out: ((image_augmentation(image_data[0], size=(image_shape[0], image_shape[1]), flip=True), image_data[1]), data_out)
            )
        # Do not flip the test images.
        else:
            ds_fused = ds_fused.map(
                lambda image_data, data_out: ((image_augmentation(image_data[0], size=(image_shape[0], image_shape[1]), flip=False), image_data[1]), data_out)
            )

        # Batch it.
        ds_fused = ds_fused.batch(batch_size)

        # Prefetch it.
        ds_fused = ds_fused.prefetch(tf.data.AUTOTUNE)

        # Cache it.
        ds_fused = ds_fused.cache()

        # Append to output list.
        ds_fused_out.append(ds_fused)

    # Return with info.
    if with_info:
        return tuple(ds_fused_out), info
    # Return without info.
    else:
        return tuple(ds_fused_out)