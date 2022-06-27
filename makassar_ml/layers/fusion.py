from __future__ import annotations
import tensorflow.keras as keras
from ..layers import (
    Patches,
    PatchEncoder,
    Time2Vec,
)


class ImageInputHead(keras.layers.Layer):
    def __init__(self,
        patch_size: int,
        num_patches: int,
        embed_dim: int,
        **kwargs,
        ):
        """Image Input head.
        """
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.patch_extractor = Patches(
            patch_size=patch_size,
        )
        self.patch_encoder = PatchEncoder(
            num_patches=num_patches,
            projection_dim=embed_dim,
        )
    
    def get_config(self) -> dict:
        """Retreive custom layer configuration for future loading.

        Returns:
            dict: Configuration dictionary.
        """
        config = super().get_config().copy()
        config.update({
            'patch_size': self.patch_size,
            'num_patches': self.num_patches,
            'embed_dim': self.embed_dim,
        })
        return config

    def call(self, x):
        x = self.patch_extractor(x)
        x = self.patch_encoder(x)
        return x


class TimeSeriesInputHead(keras.layers.Layer):
    def __init__(self,
        embed_dim: int,
        **kwargs,
        ):
        """Image Input head.
        """
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

        self.t2v = Time2Vec(embed_dim=self.embed_dim)

    def get_config(self) -> dict:
        """Retreive custom layer configuration for future loading.

        Returns:
            dict: Configuration dictionary.
        """
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
        })
        return config

    def call(self, inp):
        x = self.t2v(inp)
        x = keras.layers.Concatenate(axis=-1)([inp, x])
        return x



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