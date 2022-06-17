from __future__ import annotations
import tensorflow as tf
import tensorflow.keras as keras
from ..layers import (
    TransformerEncoderLayer,
)


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


class PatchEncoder(keras.layers.Layer):
    def __init__(self, num_patches: int, projection_dim: int):
        super().__init__()
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


def ViT(
    image_shape: tuple,
    patch_size: int,
    num_patches: int,
    embed_dim: int,
    n_class: int = None,
    n_encoders: int = 3,
    n_heads: int = 8,
    ff_dim: int = 2048,
    dropout: float = 0.0,
    fc_units: list[int] = [],
    include_top: bool = True,
    ):
    """Vision Transformer (ViT) for image classification tasks.

    Based on the paper by Alexey Dosovitskiy, et al.
    https://arxiv.org/abs/2010.11929
    """
    # Cannot have empty class number and top classification layer.
    assert not (n_class is None and include_top)

    # Input tensor.
    inp_image = keras.Input(shape=image_shape)

    # Create patches.
    x = Patches(patch_size)(inp_image)

    # Encode patches.
    x = PatchEncoder(
        num_patches=num_patches,
        projection_dim=embed_dim,
    )(x)

    # Pass image feature maps through encoders.
    for _ in range(n_encoders):
        x = TransformerEncoderLayer(
            model_dim=embed_dim,
            key_dim=None,
            n_heads=n_heads,
            ff_dim=ff_dim,
            value_dim=None,
            dropout=dropout,
            norm_type='layer',
        )(x)

    # Include classifier on top of the Transfomer Encoders.
    if include_top:

        # Flatten to (batch,num_feature_maps*embed_dim)
        x = keras.layers.Flatten(data_format='channels_last')(x)

        # Add intermediate dense layers with ReLU activation.
        if fc_units:
            for units in fc_units:
                x = keras.layers.Dense(units=units, activation='relu')(x)

        # Classifier on the end.
        x = keras.layers.Dense(units=n_class, activation='softmax')(x)

    return keras.models.Model(inputs=inp_image, outputs=x)