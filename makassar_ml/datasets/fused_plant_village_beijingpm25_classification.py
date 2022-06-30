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
            df_timeseries_range = df_timeseries_datetimeindex.loc[start_date:end_date]
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
    split: list[str, str, str],
    shuffle_files: bool,
    batch_size: int,
    ) -> tuple[tf.data.Dataset,tf.data.Dataset,tf.data.Dataset]:

    assert len(split) == 3
    assert all(isinstance(item, str) for item in split)

    # Load image dataset.
    # ds_images_train, ds_images_val, ds_images_test = tfds.load(
    ds_images_tuple = tfds.load(
        name='plant_village',
        split=split,
        shuffle_files=shuffle_files,
        as_supervised=True,
        with_info=False,
    )

    # Load weather dataset.
    df_timeseries = load_beijingpm25_df(
        path=timeseries_path,
    )

    # Fuse weather data for each split.
    ds_fused_out = []
    for i in range(len(ds_images_tuple)):

        # Randomly select dates to associate with each element.
        n_images = int(ds_images_tuple[i].cardinality())
        random_dates = df_timeseries.iloc[timeseries_reserve_offset_index:].sample(n=n_images, replace=True)[timeseries_datetime_column]

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
            df_timeseries=df_timeseries,
            timedelta_dict=timeseries_timedelta_dict,
            features=timeseries_features,
        )

        # Augment the images.
        ds_fused = ds_fused.map(
            lambda image_data, label: ((image_augmentation(image_data[0], size=(image_shape[0], image_shape[1])), image_data[1]), label)
        )

        # Batch it.
        ds_fused = ds_fused.batch(batch_size)

        # Append to output list.
        ds_fused_out.append(ds_fused)

    return tuple(ds_fused_out)