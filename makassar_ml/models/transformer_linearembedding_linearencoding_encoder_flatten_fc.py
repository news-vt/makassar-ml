from __future__ import annotations
import tensorflow as tf
import tensorflow.keras as keras
from ..layers import TransformerEncoderLayer, LinearEmbedding2D, LinearEncoding2D


def build_model(
    in_seq_len: int,
    in_feat: int,
    out_feat: int,
    fc_units: list[int], # list of fully-connected dimensions before classifier.
    embed_dim: int,
    n_heads: int,
    key_dim: int = None,
    value_dim: int = None,
    ff_dim: int = 2048,
    dropout: float = 0.0,
    n_encoders: int = 3,
    ) -> keras.Model:

    # Input sequence of features.
    inp = keras.Input(shape=(in_seq_len, in_feat))
    # Create common model input/output variable.
    x = inp
    # Time embedding.
    x = LinearEmbedding2D(embed_dim=embed_dim)(x)
    x = LinearEncoding2D()(x) # Positional encoding.
    # Pass combined featured through cascaded self-attention encoder sublayers.
    for _ in range(n_encoders):
        x = TransformerEncoderLayer(
            model_dim=embed_dim,
            key_dim=key_dim,
            n_heads=n_heads,
            ff_dim=ff_dim,
            value_dim=value_dim,
            dropout=dropout,
        )(x)
    x = keras.layers.Flatten(data_format='channels_last')(x) # Flatten to 1D (in_seq_len*embed_dim,)
    # Fully-connected network before classifier.
    for units in fc_units: 
        x = keras.layers.Dense(units=units, activation='relu')(x)
        x = keras.layers.Dropout(rate=dropout)(x)
    # Classifier.
    x = keras.layers.Dense(units=out_feat, activation='linear')(x)

    # Construct model class and return.
    return keras.Model(inputs=inp, outputs=x)
