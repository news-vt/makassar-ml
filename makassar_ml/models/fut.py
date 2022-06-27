from __future__ import annotations
import tensorflow as tf
import tensorflow.keras as keras
from ..layers import (
    TransformerEncoderLayer,
)


def FuT(
    inputs: list[tf.Tensor],
    input_heads: list[keras.layers.Layer],
    task_head: keras.layers.Layer,
    n_heads: int = 8,
    key_dim: int = None,
    value_dim: int = None,
    ff_dim: int = 2048,
    dropout: float = 0.0,
    n_encoders: int = 3,
    ) -> keras.Model:
    """Fusion Transformer (FuT).

    Combines multiple input data sources into a single task-dependent head.
    """

    # Create branch for each input sequence.
    outputs = []
    for i, (inp, inp_head) in enumerate(zip(inputs, input_heads)):
        layers = []

        # Pass input through head.
        x = inp_head(inp)

        # Preserve input dimension.
        model_dim = x.shape[-1]

        # Build encoder pipeline.
        layers.extend([
            TransformerEncoderLayer(
                model_dim=model_dim,
                key_dim=key_dim,
                n_heads=n_heads,
                ff_dim=ff_dim,
                value_dim=value_dim,
                dropout=dropout,
                norm_type='layer',
            )
            for _ in range(n_encoders)
        ])

        # Build branch.
        branch = keras.Sequential(layers=layers, name=f"branch_{i}")

        # Save outputs.
        x = branch(x)
        outputs.append(x)

    # Pass branch outputs to final task head.
    o = task_head(outputs)

    # Construct model class and return.
    return keras.Model(inputs=inputs, outputs=o)
