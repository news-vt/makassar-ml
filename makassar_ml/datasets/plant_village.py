from __future__ import annotations
import tensorflow as tf
import tensorflow_datasets as tfds


def image_augmentation(
    image: tf.Tensor,
    size: tuple[int,int] = None,
    flip: bool = True,
    ):
    if size is not None:
        # Preserve original data type for casting after resize operation.
        dtype = image.dtype
        image = tf.image.resize(image, size=size)
        image = tf.cast(image, dtype)
    if flip:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
    image = tf.cast(image, dtype=tf.float32) / tf.constant(256, dtype=tf.float32) # Rescale.
    # image = tfa.image.rotate(image, tf.random.normal(shape=[])*np.pi/180., interpolation='bilinear')
    return image


def load_data(
    image_shape: tuple[int,int,int],
    split: list[str, str, str],
    shuffle_files: bool,
    batch_size: int,
    ):
    assert len(split) == 3
    assert all(isinstance(item, str) for item in split)

    ds_train, ds_val, ds_test = tfds.load(
        name='plant_village',
        split=split,
        shuffle_files=shuffle_files,
        as_supervised=True,
        with_info=False,
    )

    # Augment the images.
    ds_train = ds_train.map(
        lambda x, y: (image_augmentation(x, size=image_shape[:2], flip=True), y)
    )
    ds_val = ds_val.map(
        lambda x, y: (image_augmentation(x, size=image_shape[:2], flip=True), y)
    )
    ds_test = ds_test.map(
        lambda x, y: (image_augmentation(x, size=image_shape[:2], flip=False), y)
    )

    # Batch the images.
    ds_train = ds_train.batch(batch_size)
    ds_val = ds_val.batch(batch_size)
    ds_test = ds_test.batch(batch_size)

    return ds_train, ds_val, ds_test
