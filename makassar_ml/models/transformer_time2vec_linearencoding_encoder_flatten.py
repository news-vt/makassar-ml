from __future__ import annotations
import tensorflow as tf
import tensorflow.keras as keras
from ..layers import Time2Vec, TransformerEncoderLayer, LinearEmbedding2D, LinearEncoding2D


def build_model(
    in_seq_len: int,
    in_feat: int,
    out_feat: int,
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
    x = Time2Vec(embed_dim=embed_dim)(x)
    # Linear positional encoding.
    x = LinearEncoding2D()(x)
    # Combine input with embedding to form attention input features.
    x = keras.layers.Concatenate(axis=-1)([inp, x])
    # Compute model dimension for Transformer encoder.
    model_dim: int = x.shape[-1]
    # Pass combined featured through cascaded self-attention encoder sublayers.
    for _ in range(n_encoders):
        x = TransformerEncoderLayer(
            model_dim=model_dim,
            key_dim=key_dim,
            n_heads=n_heads,
            ff_dim=ff_dim,
            value_dim=value_dim,
            dropout=dropout,
        )(x)
    x = keras.layers.Flatten(data_format='channels_last')(x) # Flatten to 1D (in_seq_len*model_dim,)
    # Classifier.
    x = keras.layers.Dense(units=out_feat, activation='linear')(x)

    # Construct model class and return.
    return keras.Model(inputs=inp, outputs=x)