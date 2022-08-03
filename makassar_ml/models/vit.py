from __future__ import annotations
import tensorflow as tf
import tensorflow.keras as keras
from ..layers import (
    TransformerEncoderLayer,
    Patches,
    PatchEncoder,
)


def ViT(
    image_shape: tuple,
    patch_size: int,
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
    num_patches = (image_shape[0]//patch_size)**2
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