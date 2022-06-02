from __future__ import annotations
import tensorflow as tf
import tensorflow.keras as keras


class LinearEncoding2D(keras.layers.Layer):
    """Linear encoding on 2D data.

    Encoding formulation is:
        u = x + W

    Where shapes are:
        - x (batch, seq, feat)
        - W (embed_dim, feat)
        - u (batch, seq, feat)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape: list[int]):
        self.w = self.add_weight(
            name='w',
            shape=(input_shape[1], input_shape[2]),
            initializer='uniform',
            trainable=True,
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Embed input into linear and periodic feature components.

        Args:
            x (tf.Tensor): Input tensor with shape (seq, feat)

        Returns:
            tf.Tensor: Output tensor with shape (seq, embed_dim)
        """
        return x + self.w

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Determines the output shape for a given input shape.

        Args:
            input_shape (tf.TensorShape): Input shape (seq, feat).

        Returns:
            tf.TensorShape: Output shape (seq, embed_dim).
        """
        return input_shape