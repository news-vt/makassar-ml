import tensorflow as tf
import tensorflow.keras as keras


@keras.utils.register_keras_serializable()
class Time2Vec(keras.layers.Layer):
    def __init__(self, embed_dim: int, activation: str = 'sin', **kwargs):
        """Vector embedding representation of time.

        Based on the original concept proposed by Kazemi et al., 2019 (https://arxiv.org/abs/1907.05321).

        This layer operates on a single time step with N feature dimensions. When using this layer for multi-time-step
        datasets, you must pass this layer through a `keras.layers.TimeDistributed` layer to multiplex this for all time steps.

        Note that embedding is done on a per-feature basis. For example, using an input record with 7 features (i.e., shape=(1, 7))
        and an embeddding dimension of 10, the resulting embedding would have 70 dimensions (i.e., shape=(1, 70)). This is because
        each of the 7 features gets a 10-dimensional embedding.

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

    def build(self, input_shape: list[int]):

        # Weight and bias term for linear portion (i = 0)
        # of embedding.
        self.w_linear = self.add_weight(
            name='w_linear',
            shape=(input_shape[1],),
            initializer='uniform',
            trainable=True,
        )
        self.b_linear = self.add_weight(
            name='b_linear',
            shape=(input_shape[1],),
            initializer='uniform',
            trainable=True,
        )

        # Weight and bias terms for the periodic
        # portion (1 <= i <= k) of embedding.
        self.w_periodic = self.add_weight(
            name='w_periodic',
            shape=(1, input_shape[1], self.embed_dim-1,),
            initializer='uniform',
            trainable=True,
        )
        self.b_periodic = self.add_weight(
            name='b_periodic',
            shape=(1, input_shape[1], self.embed_dim-1,),
            initializer='uniform',
            trainable=True,
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Embed input into linear and periodic feature components.

        Args:
            x (tf.Tensor): Input tensor with shape (sequence_length, feature_size)

        Returns:
            tf.Tensor: Output tensor with shape (sequence_length, feature_size * embed_dim)
        """
        # Linear term (i = 0).
        embed_linear = self.w_linear * x + self.b_linear
        embed_linear = tf.expand_dims(embed_linear, axis=-1) # Reshape to (sequence_length, feature_size, 1)

        # Periodic terms (1 <= i <= k).
        inner = keras.backend.dot(x, self.w_periodic) + self.b_periodic
        embed_periodic = self.activation_func(inner) # (sequence_length, feature_size, embed_dim - 1)

        # Return concatenated linear and periodic features.
        ret = tf.concat([embed_linear, embed_periodic], axis=-1) # (sequence_length, feature_size, embed_dim)
        ret = tf.reshape(ret, (-1, x.shape[1]*self.embed_dim)) # (sequence_length, feature_size * embed_dim)
        return ret

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Determines the output shape for a given input shape.

        Args:
            input_shape (tf.TensorShape): Input shape (sequence_length, feature_size).

        Returns:
            tf.TensorShape: Output shape (sequence_length, feature_size * embed_dim).
        """
        return tf.TensorShape((input_shape[0], input_shape[1]*self.embed_dim))

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


# # Update custom objects dictionary.
# keras.utils.get_custom_objects()['Time2Vec'] = Time2Vec



# stock_feat = 7
# seq_len = 128
# embed_dim = 10
# inp = keras.Input(shape=(seq_len, stock_feat))
# logger.info(f"{inp.shape=}")
# x = keras.layers.TimeDistributed(Time2Vec(embed_dim))(inp)
# logger.info(f"{x.shape=}")
# x = keras.layers.Concatenate(axis=-1)([inp, x])
# logger.info(f"{x.shape=}")