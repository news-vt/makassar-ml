from __future__ import annotations
import tensorflow as tf
import tensorflow.keras as keras


class LinearEmbedding2D(keras.layers.Layer):
    """Linear embedding on 2D data.

    Embedding formulation is:
        u = W * x + b

    Where shapes are:
        - x (batch, seq, feat)
        - W (embed_dim, feat)
        - b (embed_dim)
        - u (batch, seq, embed_dim)

    Args:
        embed_dim (int): Length of the embedding vector.
    """
    def __init__(self, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape: list[int]):
        # Weight and bias terms.
        self.w = self.add_weight(
            name='w',
            shape=(self.embed_dim, input_shape[-1],),
            initializer='uniform',
            trainable=True,
        )
        self.b = self.add_weight(
            name='b',
            shape=(self.embed_dim,),
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
        return tf.matmul(x, self.w, transpose_b=True) + self.b

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Determines the output shape for a given input shape.

        Args:
            input_shape (tf.TensorShape): Input shape (seq, feat).

        Returns:
            tf.TensorShape: Output shape (seq, embed_dim).
        """
        return tf.TensorShape((input_shape[1], self.embed_dim))

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