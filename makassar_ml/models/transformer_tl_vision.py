from __future__ import annotations
import tensorflow as tf
import tensorflow.keras as keras
from ..layers import (
    TransformerEncoderLayer,
)

# Vision Transformer guide.
# https://keras.io/examples/vision/image_classification_with_vision_transformer/

def build_model(
    image_shape: tuple,
    image_backbone_name: str, # Must be within `keras.applications`
    image_encoder_dim: int,
    n_class: int,
    dropout: float = 0.0,
    n_image_encoders: int = 3,
    n_image_heads: int = 8,
    image_ff_dim: int = 2048,
    fc_units: list[int] = [],
    ):
    # Ensure both lower and normal case spellings exist for method retrieval.
    assert hasattr(keras.applications, image_backbone_name.lower())
    assert hasattr(keras.applications, image_backbone_name)

    # Image feature extraction network.
    # Pre-trained on ImageNet dataset.
    image_backbone_net = getattr(keras.applications, image_backbone_name)(
        weights='imagenet',
        input_shape=image_shape,
        pooling=None,
        include_top=False,
    )
    image_backbone_net.trainable = False # Freeze the base model so that it will not be updated during training

    # Image input branch.
    inp_image = keras.Input(shape=image_shape)
    # Set output variable to current input.
    x_image = inp_image
    # Preprocess image input.
    x_image = getattr(keras.applications, image_backbone_name.lower()).preprocess_input(x_image)
    # Run preprocessed image through backbone image feature extraction network.
    x_image = image_backbone_net(x_image, training=False)
    # Project backbone output into dimension necessary for encoder layer.
    # Reduces channel dimension of image activation map so that it matches that of the encoder.
    # Shape is (batch,w,h,image_encoder_dim)
    x_image = keras.layers.Conv2D(filters=image_encoder_dim, kernel_size=(1,1))(x_image)
    # Squeeze feature map dimensions to (batch,w*h,image_encoder_dim)
    x_image = keras.layers.Reshape(target_shape=(-1,x_image.shape[-1]))(x_image)
    # Create learned positional embedding and add to feature map encoding.
    num_feature_maps = x_image.shape[1]
    pos = tf.range(
        start=0,
        limit=num_feature_maps,
        delta=1,
    )
    x_image_pos = keras.layers.Embedding(
        input_dim=num_feature_maps,
        output_dim=image_encoder_dim,
    )(pos)
    x_image = x_image + x_image_pos
    # Pass image feature maps through encoders.
    for _ in range(n_image_encoders):
        x_image = TransformerEncoderLayer(
            model_dim=image_encoder_dim,
            key_dim=None,
            n_heads=n_image_heads,
            ff_dim=image_ff_dim,
            value_dim=None,
            dropout=dropout,
            norm_type='layer',
        )(x_image)

    # Flatten to (batch,num_feature_maps*image_encoder_dim)
    x_image = keras.layers.Flatten(data_format='channels_last')(x_image)

    # Add intermediate dense layers with ReLU activation.
    for units in fc_units:
        x_image = keras.layers.Dense(units=units, activation='relu')(x_image)

    # Classifier on the end.
    x_image = keras.layers.Dense(units=n_class, activation='softmax')(x_image)

    return keras.models.Model(inputs=inp_image, outputs=x_image)