from __future__ import annotations
import tensorflow.keras as keras
from tensorflow import Tensor
from ..layers import (
    Patches,
    PatchEncoder,
    Time2Vec,
)


def ImageInputHead(
    shape: tuple,
    patch_size: int,
    num_patches: int,
    embed_dim: int,
    ) -> keras.Model:
    """Image input head."""
    # Create input tensor.
    inp = keras.layers.Input(shape=shape)
    x = inp

    # Create patches.
    x = Patches(
        patch_size=patch_size,
    )(x)

    # Encode patches.
    x = PatchEncoder(
        num_patches=num_patches,
        projection_dim=embed_dim,
    )(x)
    return keras.Model(inputs=inp, outputs=x)


def TimeSeriesInputHead(
    shape: tuple,
    embed_dim: int,
    ) -> keras.Model:
    """Time-Series input head."""
    # Create input tensor.
    inp = keras.layers.Input(shape=shape)
    x = inp

    # Time-vector embedding.
    x = Time2Vec(embed_dim=embed_dim)(x)

    # Combine input with embedding to form input features.
    x = keras.layers.Concatenate(axis=-1)([inp, x])
    return keras.Model(inputs=inp, outputs=x)



class ClassificationTaskHead(keras.layers.Layer):
    def __init__(self, 
        n_class: int,
        fc_units: list[int] = [],
        dropout: float = 0.0,
        **kwargs,
        ):
        """Classification Task Head.
        """
        super().__init__(**kwargs)
        assert isinstance(n_class, int)
        assert isinstance(fc_units, (tuple, list))
        assert isinstance(dropout, (int, float))
        self.n_class = n_class
        self.fc_units = fc_units
        self.dropout = dropout

        # Fully-connected.
        layers = []
        for units in self.fc_units:
            layers.append(keras.layers.Dense(units=units, activation='relu'))
            layers.append(keras.layers.Dropout(rate=self.dropout))

        # Classifier.
        layers.append(keras.layers.Dense(units=self.n_class, activation='softmax'))

        # Build dense network.
        self.dense = keras.Sequential(layers=layers)

    def call(self, inputs):

        # Flatten the input branches.
        branches = []
        for i, inp in enumerate(inputs):
            x = keras.layers.Flatten(data_format='channels_last')(inp)
            branches.append(x)

        # Concatenate.
        x = keras.layers.Concatenate(axis=-1)(branches)

        # Pass through dense network.
        x = self.dense(x)
        return x

    def get_config(self) -> dict:
        """Retreive custom layer configuration for future loading.

        Returns:
            dict: Configuration dictionary.
        """
        config = super().get_config().copy()
        config.update({
            'n_class': self.n_class,
            'fc_units': self.fc_units,
            'dropout': self.dropout,
        })
        return config



# Update custom objects dictionary.
keras.utils.get_custom_objects()['ImageInputHead'] = ImageInputHead
keras.utils.get_custom_objects()['TimeSeriesInputHead'] = TimeSeriesInputHead
keras.utils.get_custom_objects()['ClassificationTaskHead'] = ClassificationTaskHead