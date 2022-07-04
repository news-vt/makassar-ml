from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from .beijingpm25 import load_beijingpm25_df
from .plant_village import image_augmentation


def fuse_dated_images_timeseries(
    ds_images_dates: tf.data.Dataset,
    df_timeseries: pd.DataFrame,
    timedelta_dict: dict, # i.e., {'days':1, 'hours':24, 'minutes':5}
    datetime_column: str = 'datetime',
    datetime_format: str = '%Y-%m-%d %H:%M:%S',
    features: list[str] = None,
    ):

    # Set the index to datetime so lookup is quicker.
    df_timeseries_datetimeindex = df_timeseries.set_index([datetime_column])

    def gen():
        for (image, date), label in ds_images_dates:
            end_date = dt.datetime.strptime(date.numpy().decode('utf8'), datetime_format)
            start_date = end_date - dt.timedelta(**timedelta_dict)
            df_timeseries_range = df_timeseries_datetimeindex.loc[start_date:end_date].iloc[:-1] # Exclude the last one. WARNING: this assumes that the end timestamp is in the dataset.
            df_timeseries_range.reset_index(inplace=True)
            if features is not None:
                df_timeseries_range = df_timeseries_range[features]
            tensor_timeseries_range = tf.convert_to_tensor(df_timeseries_range)
            yield ((image, tensor_timeseries_range), label)

    ds_fused = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            (
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8), # image
                tf.TensorSpec(shape=(None, None), dtype=tf.float32), # timeseries
            ),
            tf.TensorSpec(shape=(), dtype=tf.int64), # label
        ),
    )
    # Set length of dataset since it will be the same as the image dataset.
    ds_fused = ds_fused.apply(tf.data.experimental.assert_cardinality(tf.data.experimental.cardinality(ds_images_dates)))
    return ds_fused


def load_data(
    timeseries_path: str,
    timeseries_reserve_offset_index: int,
    timeseries_datetime_column: str,
    timeseries_features: list[str],
    timeseries_timedelta_dict: dict,
    image_shape: list[int, int, int],
    split: list[float, float, float],
    shuffle_files: bool,
    batch_size: int,
    with_info: bool = False,
    norm: bool = True,
    ) -> tuple[tf.data.Dataset,tf.data.Dataset,tf.data.Dataset]:

    assert len(split) == 3
    assert all(isinstance(item, float) for item in split)
    assert np.isclose(sum(split), 1.0)

    # Load image dataset.
    split_images = [
        f"train[0%:{int(split[0]*100)}%]",
        f"train[{int(split[0]*100)}%:{100-int(split[-1]*100)}%]",
        f"train[{100-int(split[-1]*100)}%:]",
    ]
    ds_images_tuple, info = tfds.load(
        name='plant_village',
        split=split_images,
        shuffle_files=shuffle_files,
        as_supervised=True,
        with_info=True,
    )

    # Load weather dataset.
    df_timeseries_tuple = load_beijingpm25_df(
        path=timeseries_path,
        split=split,
    )
    for df in df_timeseries_tuple:
        df.dropna(inplace=True) # Remove NaN.
        df.reset_index(inplace=True)

    # Normalize the time-series data.
    if norm:
        mean = df_timeseries_tuple[0][timeseries_features].mean()
        std = df_timeseries_tuple[0][timeseries_features].std()
        for i, df in enumerate(df_timeseries_tuple):
            df_timeseries_tuple[i][timeseries_features] = (df_timeseries_tuple[i][timeseries_features] - mean)/std

    # Fuse weather data for each split.
    ds_fused_out = []
    for i in range(len(ds_images_tuple)):

        # Randomly select dates to associate with each element.
        n_images = int(ds_images_tuple[i].cardinality())
        random_dates = df_timeseries_tuple[i].iloc[timeseries_reserve_offset_index:].sample(n=n_images, replace=True)[timeseries_datetime_column]

        # Fuse images with dates.
        ds_images_dates = tf.data.Dataset.zip(
            (ds_images_tuple[i], tf.data.Dataset.from_tensor_slices(random_dates.astype(str)))
        ).map(
            lambda image_label, datestring: ((image_label[0], datestring), image_label[1]) # Reorder to ((image, date), label).
        )

        # Fuse the datasets.
        # Returns ((image, timeseries), label)
        ds_fused = fuse_dated_images_timeseries(
            ds_images_dates=ds_images_dates, 
            df_timeseries=df_timeseries_tuple[i],
            timedelta_dict=timeseries_timedelta_dict,
            features=timeseries_features,
        )

        # Augment the images.
        # Flip the train/val.
        if i < 2:
            ds_fused = ds_fused.map(
                lambda image_data, label: ((image_augmentation(image_data[0], size=(image_shape[0], image_shape[1]), flip=True), image_data[1]), label)
            )
        # Do not flip the test images.
        else:
            ds_fused = ds_fused.map(
                lambda image_data, label: ((image_augmentation(image_data[0], size=(image_shape[0], image_shape[1]), flip=False), image_data[1]), label)
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