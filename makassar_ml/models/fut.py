from __future__ import annotations
import tensorflow as tf
import tensorflow.keras as keras
from ..layers import (
    TransformerEncoderLayer,
    ImageInputHead,
    TimeSeriesInputHead,
    ClassificationTaskHead,
    RegressionTaskHead,
)


def FuT(
    inputs: list[keras.Model|tf.Tensor],
    task_head: keras.layers.Layer|list[keras.layers.Layer],
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
    if isinstance(task_head, keras.layers.Layer):
        task_heads = [task_head]
    else:
        task_heads = task_head
    outputs = []
    for head in task_heads:
        o = head(io_outputs)
        outputs.append(o)

    # Construct model class and return.
    return keras.Model(inputs=io_inputs, outputs=outputs)



def FuT_image_timeseries_classifier(
    image_shape: tuple,
    seq_shape: tuple,
    patch_size: int,
    image_embed_dim: int,
    seq_embed_dim: int,
    n_class: int,
    num_patches: int = None,
    n_heads: int = 8,
    key_dim: int = None,
    value_dim: int = None,
    ff_dim: int = 2048,
    fc_units: list[int] = [],
    dropout: float = 0.0,
    n_encoders: int = 3,
    ) -> keras.Model:
    """Vision Forecast Transformer for classification tasks."""
    
    if num_patches is None:
        num_patches = (image_shape[0]//patch_size)**2

    # Input heads.
    inputs = [
        ImageInputHead(
            shape=image_shape,
            patch_size=patch_size,
            num_patches=num_patches,
            embed_dim=image_embed_dim,
        ),
        TimeSeriesInputHead(
            shape=seq_shape,
            embed_dim=seq_embed_dim,
        ),
    ]

    # Task head.
    task_head = ClassificationTaskHead(
        n_class=n_class,
        dropout=dropout,
        fc_units=fc_units,
        name='classifier',
    )

    # Create fusion model.
    model = FuT(
        inputs=inputs,
        task_head=task_head,
        n_heads=n_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        ff_dim=ff_dim,
        dropout=dropout,
        n_encoders=n_encoders,
    )
    return model


def FuT_image_timeseries_regression(
    image_shape: tuple,
    in_seq_shape: tuple,
    out_seq_shape: tuple,
    patch_size: int,
    image_embed_dim: int,
    seq_embed_dim: int,
    fusion_embed_dim: int,
    num_patches: int = None,
    n_heads: int = 8,
    key_dim: int = None,
    value_dim: int = None,
    ff_dim: int = 2048,
    dropout: float = 0.0,
    n_encoders: int = 3,
    ) -> keras.Model:
    """Vision Forecast Transformer for regression tasks."""
    
    if num_patches is None:
        num_patches = (image_shape[0]//patch_size)**2

    # Input heads.
    inputs = [
        ImageInputHead(
            shape=image_shape,
            patch_size=patch_size,
            num_patches=num_patches,
            embed_dim=image_embed_dim,
        ),
        TimeSeriesInputHead(
            shape=in_seq_shape,
            embed_dim=seq_embed_dim,
        ),
    ]

    # Task head.
    task_head = RegressionTaskHead(
        out_seq_shape=out_seq_shape,
        embed_dim=fusion_embed_dim,
        name='regressor',
    )

    # Create fusion model.
    model = FuT(
        inputs=inputs,
        task_head=task_head,
        n_heads=n_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        ff_dim=ff_dim,
        dropout=dropout,
        n_encoders=n_encoders,
    )
    return model


def FuT_image_timeseries_multitask(
    image_shape: tuple,
    in_seq_shape: tuple,
    out_seq_shape: tuple,
    n_class: int,
    patch_size: int,
    image_embed_dim: int,
    seq_embed_dim: int,
    fusion_embed_dim: int,
    num_patches: int = None,
    n_heads: int = 8,
    key_dim: int = None,
    value_dim: int = None,
    ff_dim: int = 2048,
    fc_units: list[int] = [],
    dropout: float = 0.0,
    n_encoders: int = 3,
    ) -> keras.Model:
    """Vision Forecast Transformer for regression tasks."""
    
    if num_patches is None:
        num_patches = (image_shape[0]//patch_size)**2

    # Input heads.
    inputs = [
        ImageInputHead(
            shape=image_shape,
            patch_size=patch_size,
            num_patches=num_patches,
            embed_dim=image_embed_dim,
        ),
        TimeSeriesInputHead(
            shape=in_seq_shape,
            embed_dim=seq_embed_dim,
        ),
    ]

    # Task head.
    task_heads = [
        ClassificationTaskHead(
            n_class=n_class,
            dropout=dropout,
            fc_units=fc_units,
            name='classifier',
        ),
        RegressionTaskHead(
            out_seq_shape=out_seq_shape,
            embed_dim=fusion_embed_dim,
            name='regressor',
        ),
    ]

    # Create fusion model.
    model = FuT(
        inputs=inputs,
        task_head=task_heads,
        n_heads=n_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        ff_dim=ff_dim,
        dropout=dropout,
        n_encoders=n_encoders,
    )
    return model