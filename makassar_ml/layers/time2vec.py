import tensorflow as tf
import tensorflow.keras as keras


class Time2Vec(keras.layers.Layer):
    def __init__(self, embed_dim: int, activation: str = 'sin', **kwargs):
        """Vector embedding representation of time.

        Based on the original concept proposed by Kazemi et al., 2019 (https://arxiv.org/abs/1907.05321).

        Input is assumed to have dimension `(batch,seq,feat)`.

        Args:
            embed_dim (int): Length of the time embedding vector.
            activation (str, optional): Periodic activation function. Possible values are ['sin', 'cos']. Defaults to 'sin'.
        """
        super().__init__(**kwargs)
        self.embed_dim = embed_dim # Embed dimension is k+1.
        self.activation = activation.lower() # Convert to lower-case.

        # Set periodic activation function.
        if self.activation.startswith('sin'):
            self.activation_func = tf.sin
        elif self.activation.startswith('cos'):
            self.activation_func = tf.cos
        else:
            raise ValueError(f'Unsupported periodic activation function "{activation}"')

    def build(self, input_shape: tf.TensorShape):
        """Creates the variables of the layer.

        Args:
            input_shape (tf.TensorShape): Shape of the input.
        """
        # Embedding length must be longer than feature dimension.
        assert self.embed_dim > input_shape[-1]

        # Weight and bias term for linear portion (i = 0)
        # of embedding.
        self.w_linear = self.add_weight(
            name='w_linear',
            shape=(input_shape[1],1,),
            initializer='uniform',
            trainable=True,
        )
        self.b_linear = self.add_weight(
            name='b_linear',
            shape=(input_shape[1],1,),
            initializer='uniform',
            trainable=True,
        )

        # Weight and bias terms for the periodic
        # portion (1 <= i <= k) of embedding.
        self.w_periodic = self.add_weight(
            name='w_periodic',
            shape=(input_shape[-1],self.embed_dim-input_shape[-1],), # (feat,embed_dim-feat)
            initializer='uniform',
            trainable=True,
        )
        self.b_periodic = self.add_weight(
            name='b_periodic',
            shape=(input_shape[1],self.embed_dim-input_shape[-1],), # (seq,embed_dim-feat)
            initializer='uniform',
            trainable=True,
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Embed input into linear and periodic feature components.

        Args:
            x (tf.Tensor): Input tensor with shape `(batch,seq,feat)`.

        Returns:
            tf.Tensor: Output tensor with shape `(batch,seq,embed_dim)`.
        """
        # Linear terms.
        embed_linear = self.w_linear * x + self.b_linear # (batch,seq,feat)

        # Periodic terms.
        inner = tf.matmul(x, self.w_periodic) + self.b_periodic
        embed_periodic = self.activation_func(inner) # (batch,seq,embed_dim-feat)

        # Return concatenated linear and periodic features.
        ret = tf.concat([embed_linear, embed_periodic], axis=-1) # (batch,seq,embed_dim)
        return ret

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Determines the output shape for a given input shape.

        Args:
            input_shape (tf.TensorShape): Input shape `(batch,seq,feat)`.

        Returns:
            tf.TensorShape: Output shape `(batch,seq,embed_dim)`.
        """
        return tf.TensorShape((input_shape[0], input_shape[1], self.embed_dim))

    def get_config(self) -> dict:
        """Retreive custom layer configuration for future loading.

        Returns:
            dict: Configuration dictionary.
        """
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'activation': self.activation,
        })
        return config


# Update custom objects dictionary.
keras.utils.get_custom_objects()['Time2Vec'] = Time2Vec



# stock_feat = 7
# seq_len = 128
# embed_dim = 10
# inp = keras.Input(shape=(seq_len, stock_feat))
# logger.info(f"{inp.shape=}")
# x = keras.layers.TimeDistributed(Time2Vec(embed_dim))(inp)
# logger.info(f"{x.shape=}")
# x = keras.layers.Concatenate(axis=-1)([inp, x])
# logger.info(f"{x.shape=}")