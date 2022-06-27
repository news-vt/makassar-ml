from __future__ import annotations
import tensorflow as tf
import tensorflow.keras as keras
from ..layers import (
    TransformerEncoderLayer,
)


def FuT(
    inputs: list[keras.Model|tf.Tensor],
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
    io_inputs = []
    io_outputs = []
    for i, inp in enumerate(inputs):
        layers = []

        # Preserve original input tensor.
        if isinstance(inp, keras.Model):
            io_inputs.append(inp.input)
            x = inp.output
        else:
            io_inputs.append(inp)
            x = inp

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
        io_outputs.append(x)

    # Pass branch outputs to final task head.
    o = task_head(io_outputs)

    # Construct model class and return.
    return keras.Model(inputs=io_inputs, outputs=o)
