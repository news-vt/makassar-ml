from __future__ import annotations
import tensorflow.keras as keras
from ..layers import (
    TransformerEncoderLayer,
    Time2Vec,
)


def FoT(
    in_seq_len: int,
    in_feat: int,
    embed_dim: int,
    out_feat: int = None,
    n_heads: int = 8,
    key_dim: int = None,
    value_dim: int = None,
    ff_dim: int = 2048,
    dropout: float = 0.0,
    n_encoders: int = 3,
    include_top: bool = True,
    ) -> keras.Model:
    """Forecast Transformer (FoT).

    Uses Time2Vec input feature embedding proposed by Kazemi et al., 2019 (https://arxiv.org/abs/1907.05321).
    """

    # Cannot have empty output feature number and top regressor layer.
    assert not (out_feat is None and include_top)

    # Input sequence of features.
    inp = keras.Input(shape=(in_seq_len, in_feat))
    # Create common model input/output variable.
    x = inp
    x = Time2Vec(embed_dim=embed_dim)(x)
    # Combine input with embedding to form attention input features.
    x = keras.layers.Concatenate(axis=-1)([inp, x])
    # Set model dimension, since embedding dimension could be dynamic.
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
            norm_type='layer',
        )(x)

    # Include regressor on top of the Transfomer Encoders.
    if include_top:
        # Flatten to 1D (in_seq_len*embed_dim,)
        x = keras.layers.Flatten(data_format='channels_last')(x) 
        # Regressor.
        x = keras.layers.Dense(units=out_feat, activation='linear')(x)

    # Construct model class and return.
    return keras.Model(inputs=inp, outputs=x)
