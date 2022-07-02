from __future__ import annotations
import tensorflow as tf
import tensorflow.keras as keras


class PatchEncoder(keras.layers.Layer):
    def __init__(self, num_patches: int, projection_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim

        # Create layers.
        self.projection = keras.layers.Dense(units=projection_dim)
        self.position_embedding = keras.layers.Embedding(
            input_dim=num_patches,
            output_dim=projection_dim,
        )

        # Pre-create position embedding projection since it doesn't depend on the input.
        self.positions = tf.range(start=0, limit=self.num_patches, delta=1)

    def call(self, patch):
        encoded = self.projection(patch) + self.position_embedding(self.positions)
        return encoded

    def get_config(self) -> dict:
        """Retreive custom layer configuration for future loading.

        Returns:
            dict: Configuration dictionary.
        """
        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim,
        })
        return config

# Update custom objects dictionary.
keras.utils.get_custom_objects()['PatchEncoder'] = PatchEncoder