from __future__ import annotations
import tensorflow as tf
import tensorflow.keras as keras


class Patches(keras.layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images: tf.Tensor):
        # Get patches from the original image.
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        # Reshape while preserving batch dimension.
        patches = keras.layers.Reshape(
            target_shape=(-1,patches.shape[-1])
        )(patches)
        return patches

    def get_config(self) -> dict:
        """Retreive custom layer configuration for future loading.

        Returns:
            dict: Configuration dictionary.
        """
        config = super().get_config().copy()
        config.update({
            'patch_size': self.patch_size,
        })
        return config

# Update custom objects dictionary.
keras.utils.get_custom_objects()['Patches'] = Patches