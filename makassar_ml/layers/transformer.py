import tensorflow as tf
import tensorflow.keras as keras


@keras.utils.register_keras_serializable()
class PointWiseFeedForwardLayer(keras.layers.Layer):
    def __init__(self, dims: list[int], activation: str = 'gelu', **kwargs):
        """Generic point-wise feed forward layer subnetwork.

        Args:
            dims (list[int]): List of dense layer dimensions. The length of the list determines the number of dimensions. Must be at least 2 dimensions given.
            activation (str, optional): Activation function to use for the first `N-1` dense layers. The final layer has no activation.
        """
        super().__init__(**kwargs)
        assert len(dims) > 1 # Must provide at least 2 dimensions.
        self.dims = dims
        self.n_dim = len(self.dims)
        self.activation = activation

    def build(self, input_shape):
        self.ff_layers = []
        for i, dim in enumerate(self.dims):
            if i < self.n_dim-1:
                self.ff_layers.append(keras.layers.Dense(dim, activation=self.activation))
            else:
                self.ff_layers.append(keras.layers.Dense(dim))

    def call(self, x):
        for i in range(self.n_dim):
            x = self.ff_layers[i](x)
        return x

    def get_config(self) -> dict:
        """Retreive custom layer configuration for future loading.

        Returns:
            dict: Configuration dictionary.
        """
        config = super().get_config().copy()
        config.update({
            'dims': self.dims,
            'activation': self.activation,
        })
        return config

# # Update custom objects dictionary.
# keras.utils.get_custom_objects()['PointWiseFeedForwardLayer'] = PointWiseFeedForwardLayer

@keras.utils.register_keras_serializable()
class TransformerEncoderLayer(keras.layers.Layer):
    def __init__(self,
        model_dim: int,
        n_heads: int,
        key_dim: int = None,
        value_dim: int = None,
        ff_dim: int = 2048,
        dropout: float = 0.0,
        **kwargs,
        ):
        """Transformer encoder layer.

        Based on the original concept proposed by Vaswani et al., 2017 (https://arxiv.org/abs/1706.03762).

        Args:
            model_dim (int): Encoder input and output feature dimensions.
            n_heads (int): Number of attention heads.
            key_dim (int, optional): Key dimension. If `None` is specified then defaults to `int(model_dim/n_heads)`. Defaults to `None`.
            value_dim (int, optional): Value dimension. If None is specified the Key dimension will be used. Defaults to `None`.
            ff_dim (int, optional): Dimension of the feed forward sublayer. Defaults to `2048`.
            dropout (float, optional): Dropout rate. Defaults to `0.0`.
        """
        super().__init__(**kwargs)
        self.model_dim = model_dim
        self.n_heads = n_heads # Number of attention heads.
        if key_dim is None:
            self.key_dim = max(int(model_dim/n_heads), 1)
        else:
            self.key_dim = key_dim
        if value_dim is None:
            self.value_dim = max(int(model_dim/n_heads), 1)
        else:
            self.value_dim = value_dim
        self.ff_dim = ff_dim
        self.dropout = dropout

    def build(self, input_shape: tf.TensorShape):

        # First sublayer.
        # Multi-head attention with add and norm.
        self.attn_multi = keras.layers.MultiHeadAttention(
            num_heads=self.n_heads,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            output_shape=self.model_dim,
        )
        self.attn_multi._build_from_signature(input_shape, input_shape, input_shape)
        self.attn_dropout = keras.layers.Dropout(rate=self.dropout)
        self.attn_add = keras.layers.Add()
        # self.attn_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.attn_norm = keras.layers.BatchNormalization()

        # Second sublayer.
        # Point-wise feed forward network with add and norm.
        # d_query_feat = input_shape[0][-1] # Query feature size.
        self.ff_net = PointWiseFeedForwardLayer(
            dims=[self.ff_dim, self.model_dim],
            activation='gelu',
        )
        self.ff_dropout = keras.layers.Dropout(rate=self.dropout)
        self.ff_add = keras.layers.Add()
        # self.ff_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.ff_norm = keras.layers.BatchNormalization()

    def call(self, 
        x: tf.Tensor,
        training: bool = False,
        ) -> tf.Tensor:
        """Encode input using multi-head self-attention mechanisms.

        Args:
            x (tf.Tensor): Batched input sequence into the encoder layer with shape `(batch_size, sequence_length, model_dim)`.
            training (bool, optional): Indicates whether the `call` is meant for training or inference. Defaults to `False`.

        Returns:
            tf.Tensor: Output tensor with shape (batch_size, sequence_length, model_dim)
        """
        # First, do the attention sublayer.
        x_attn = self.attn_multi(x, x, x) # Unpack input as Query, Value, and optional Key.
        x_attn = self.attn_dropout(x_attn, training=training)
        x_attn = self.attn_add([x, x_attn]) # (residual) Add Query matrix with result of attention layer.
        x_attn = self.attn_norm(x_attn, training=training) # Normalize the residual.

        # Second, do the feed forward sublayer.
        x_ff = self.ff_net(x_attn)
        x_ff = self.ff_dropout(x_ff, training=training)
        x_ff = self.ff_add([x_attn, x_ff])
        x_ff = self.ff_norm(x_ff, training=training)

        # Return output of feed forward sublayer.
        return x_ff

    def get_config(self) -> dict:
        """Retreive custom layer configuration for future loading.

        Returns:
            dict: Configuration dictionary.
        """
        config = super().get_config().copy()
        config.update({
            'model_dim': self.model_dim,
            'n_heads': self.n_heads,
            'key_dim': self.key_dim,
            'ff_dim': self.ff_dim,
            'value_dim': self.value_dim,
            'dropout': self.dropout,
        })
        return config

# # Update custom objects dictionary.
# keras.utils.get_custom_objects()['TransformerEncoderLayer'] = TransformerEncoderLayer