from __future__ import annotations
import tensorflow as tf
import tensorflow.keras as keras


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

class TransformerDecoderLayer(keras.layers.Layer):
    def __init__(self,
        model_dim: int,
        n_heads: int,
        key_dim: int = None,
        value_dim: int = None,
        ff_dim: int = 2048,
        dropout: float = 0.0,
        **kwargs,
        ):
        """Transformer decoder layer.

        Based on the original concept proposed by Vaswani et al., 2017 (https://arxiv.org/abs/1706.03762).

        Args:
            model_dim (int): Decoder input and output feature dimensions.
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

    def build(self, input_shape: tuple[tf.TensorShape,tf.TensorShape]):
        print(f"{input_shape}")
        assert len(input_shape) == 2

        #### First sublayer ####
        # Masked multi-head self-attention with add and norm.
        self.l1_attn_multi = keras.layers.MultiHeadAttention(
            num_heads=self.n_heads,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            output_shape=self.model_dim,
        )
        # self.l1_attn_multi._build_from_signature(input_shape[0], input_shape[0], input_shape[0])
        self.l1_dropout = keras.layers.Dropout(rate=self.dropout)
        self.l1_add = keras.layers.Add()
        # self.l1_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.l1_norm = keras.layers.BatchNormalization()

        #### Second sublayer ####
        # Multi-head attention with add and norm.
        self.l2_attn_multi = keras.layers.MultiHeadAttention(
            num_heads=self.n_heads,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            output_shape=self.model_dim,
        )
        # self.l2_attn_multi._build_from_signature(input_shape[-1], input_shape[-1], self.l1_attn_multi.shape)
        self.l2_dropout = keras.layers.Dropout(rate=self.dropout)
        self.l2_add = keras.layers.Add()
        # self.l2_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.l2_norm = keras.layers.BatchNormalization()

        #### Third sublayer ####
        # Point-wise feed forward network with add and norm.
        # d_query_feat = input_shape[0][-1] # Query feature size.
        self.l3_ff = PointWiseFeedForwardLayer(
            dims=[self.ff_dim, self.model_dim],
            activation='gelu',
        )
        self.l3_dropout = keras.layers.Dropout(rate=self.dropout)
        self.l3_add = keras.layers.Add()
        # self.l3_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.l3_norm = keras.layers.BatchNormalization()

    def call(self,
        x: tuple[tf.Tensor, tf.Tensor],
        look_ahead_mask: tf.Tensor = None,
        memory_mask: tf.Tensor = None,
        training: bool = False,
        return_attention_scores: bool = False,
        ) -> tf.Tensor|tuple[tf.Tensor,tuple[tf.Tensor,tf.Tensor]]:
        """Decode input using multi-head attention.

        Accepts decoder input sequence `x` and `memory` sequence 
        (often the output of the last encoder layer).

        Args:
            x (tuple[tf.Tensor, tf.Tensor]): List of batched decoder input sequence, and batched memory sequence (often the output of the last encoder layer).
            look_ahead_mask (tf.Tensor, optional): Attention mask for the decoder input sequence. Defaults to `None`.
            memory_mask (tf.Tensor, optional): Attention mask for the memory sequence. Defaults to `None`.
            training (bool, optional): Indicates whether the `call` is meant for training or inference. Defaults to `False`.
            return_attention_scores (bool, optional): Indicates whether the output is `(attention_output, attention_scores)` if `True`, or `attention_output` if `False`. Defaults to `False`.

        Returns:
            tf.Tensor|tuple[tf.Tensor,tuple[tf.Tensor,tf.Tensor]]: Output depends on `return_attention_scores` value.  Returns `(attention_output, attention_scores)` if `return_attention_scores=True`, or `attention_output` if `return_attention_scores=False`.
        """
        # Unpack target and memory from input list.
        target, memory = x

        # Self-attention input layer.
        x_l1_attn, x_l1_attn_weights = self.l1_attn_multi(
            target, target, target,
            attention_mask=look_ahead_mask,
            return_attention_scores=True,
        )
        x_l1_attn = self.l1_dropout(x_l1_attn, training=True)
        out_l1 = self.l1_add([x_l1_attn, target])
        out_l1 = self.l1_norm(out_l1, training=training)

        # Second attention layer with previous encoder input.
        x_l2_attn, x_l2_attn_weights = self.l2_attn_multi(
            memory, memory, out_l1,
            attention_mask=memory_mask,
            return_attention_scores=True,
        )
        x_l2_attn = self.l2_dropout(x_l2_attn, training=training)
        out_l2 = self.l2_add([x_l2_attn, out_l1])
        out_l2 = self.l2_norm(out_l2, training=training)

        # Third layer with point-wise feed forward network.
        x_l3 = self.l3_ff(out_l2)
        x_l3 = self.l3_dropout(x_l3, training=training)
        out_l3 = self.l3_add([x_l3, out_l2])
        out_l3 = self.l3_norm(out_l3, training=training)

        # Return attention scores.
        if return_attention_scores:
            return out_l3, (x_l1_attn_weights, x_l2_attn_weights)
        # Return final layer output.
        else:
            return out_l3

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


# Update custom objects dictionary.
keras.utils.get_custom_objects()['PointWiseFeedForwardLayer'] = PointWiseFeedForwardLayer
keras.utils.get_custom_objects()['TransformerEncoderLayer'] = TransformerEncoderLayer
keras.utils.get_custom_objects()['TransformerDecoderLayer'] = TransformerDecoderLayer