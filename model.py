#!/usr/bin/env python
# coding: utf-8

# In[105]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection, metrics
import tensorflow as tf
from tensorflow.keras.layers import Layer, Multiply,GlobalAveragePooling1D,MultiHeadAttention,Embedding,Lambda,Dense,Flatten,Conv2D,Dropout, Conv2DTranspose, MaxPooling2D, Input, Activation, Concatenate, UpSampling2D, Resizing,Reshape,Add,LayerNormalization,BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from tensorflow.keras.saving import register_keras_serializable
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.colors as colors
from tensorflow.keras.models import load_model
#from patchify import patchify
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing import image


config = {}
config["image_size"] = 128
config["num_channels"] = 3
config["num_layers"] = 12
config["hidden_dim"] = 512
config["mlp_dim"] = 3072
config["num_heads"] = 12
config["dropout_rate"] = 0.1
#config["num_patches"] = 256
config["patch_size"] = 16
config["num_patches"] = (config["image_size"]**2) // (config["patch_size"]**2)
print(config["num_patches"])


@keras.saving.register_keras_serializable()
class Create_Patches(Layer):
    def __init__(self, patch_size, channels=3, **kwargs):
        super(Create_Patches, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.channels = channels
        # Use a Conv2D layer with filters = window_size * window_size * channels
        self.conv = Conv2D(
            filters=patch_size * patch_size * channels,  # Number of patch features
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
        )

    def call(self, images):
        # Apply convolution to extract patches
        patches = self.conv(images)
        shape = tf.shape(patches)
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]
        #B, H, W, C = tf.shape(patches)
        # Reshape to [B, H*W, C] where C = window_size * window_size * channels
        patches = tf.reshape(patches, [B, H * W, C])
        return patches

    def compute_output_shape(self, input_shape):
        B, H, W, C = input_shape
        # Calculate new height and width after convolution
        new_H = H // self.patch_size
        new_W = W // self.patch_size
        # Output shape is [B, new_H * new_W, window_size * window_size * channels]
        return (B, new_H * new_W, self.patch_size * self.patch_size * self.channels)


@keras.saving.register_keras_serializable()
class LinearEmbedding(Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(LinearEmbedding, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = Dense(projection_dim)
        self.position_embedding = Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        # calculate patches embeddings
        patches_embed = self.projection(patch)
        # calcualte positional embeddings
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        positions_embed = self.position_embedding(positions)
        # add both embeddings
        encoded = patches_embed + positions_embed
        return encoded



@keras.saving.register_keras_serializable()
class MLP(Layer):
    def __init__(self, mlp_dim, hidden_dim, dropout_rate=0.1, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.dense1 = Dense(mlp_dim, activation=tf.nn.gelu)
        self.dense2 = Dense(hidden_dim)
        self.dropout = Dropout(dropout_rate)

    def call(self, x):
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        y = self.dropout(x)
        return y



@keras.saving.register_keras_serializable()
class TransformerEncoder(Layer):
    def __init__(self, num_heads, hidden_dim, mlp_dim, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim

        self.layer_norm1 = LayerNormalization()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=hidden_dim)
        self.add1 = Add()

        self.layer_norm2 = LayerNormalization()
        self.mlp = MLP(config["mlp_dim"],config["hidden_dim"])
        self.add2 = Add()

    def call(self, inputs):
        # First sub-layer: Multi-Head Self-Attention
        skip1 = inputs
        x = self.layer_norm1(inputs)
        x = self.mha(x, x)
        x = self.add1([x, skip1])

        # Second sub-layer: Feed Forward Network (MLP)
        skip2 = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = self.add2([x, skip2])

        return x



def conv_block(x, num_filters):
    x = Conv2D(num_filters, (3, 3), padding='same', kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filters, (3, 3), padding='same', kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x



def deconv_block(x, num_filters):
    x = Conv2DTranspose(num_filters, kernel_size=2, padding="same", strides=2)(x)
    return x


def cnn_encoder(inputs, filters=[64, 128, 256, 512]):

    skips = []  # List to store skip connections
    x = inputs  # Current feature map, starting with the input

    # Apply convolutional blocks and pooling
    for f in filters:
        x = conv_block(x, f)  # Apply convolutional block with specified filters
        skips.append(x)       # Store feature map before pooling as a skip connection
        x = MaxPooling2D(pool_size=(2, 2))(x)  # Downsample by 2x

    # Bottom level: apply an additional convolutional block after the last pooling
    x = conv_block(x, 1024)  # e.g., (8, 8, 1024) for input (128, 128, 3)

    # Return the bottom level and skip connections in reverse order (deepest to shallowest)
    return x, skips[::-1]



ViT = load_model("ViT-Unet.h5") # ViT-Unet_hamal  ViT-Unet



