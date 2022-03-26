from typing import List, Tuple, Union

import tensorflow as tf


def DoubleConv(
    x: tf.Tensor,
    filters: Union[int, Tuple[int, int]] = (16, 16),
    kernel_size: int = 3,
    strides: Union[int, Tuple[int, int]] = (1, 1),
    batch_norm: bool = True,
) -> tf.Tensor:

    if type(filters) == tuple:
        f1, f2 = filters
    else:
        f1 = f2 = filters

    if type(strides) == tuple:
        s1, s2 = strides
    else:
        s1 = s2 = strides

    y = tf.keras.layers.Conv2D(
        filters=f1, kernel_size=kernel_size, strides=s1, padding='same')(x)
    if batch_norm:
        y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)

    y = tf.keras.layers.Conv2D(
        filters=f2, kernel_size=kernel_size, strides=s2, padding='same')(y)
    if batch_norm:
        y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)

    return y


def ConvTranspose(x: tf.Tensor, filters: int, conv_transpose: bool = False) -> tf.Tensor:
    if conv_transpose:
        y = tf.keras.layers.Conv2DTranspose(
            filters=filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    else:
        y = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        y = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=3, strides=1, padding='same')(y)

    return y


def UNet(
    input_shape: List[int] = (24, 32, 1),
    in_out_filters: int = 16,
    batch_norm: bool = False,
    conv_transpose: bool = False,
    squeeze: bool = False,
    double_double_conv: bool = False,
) -> tf.keras.Model:

    x = tf.keras.Input(shape=input_shape)

    y1 = DoubleConv(x, filters=in_out_filters, kernel_size=3, strides=1, batch_norm=batch_norm)

    y2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(y1)
    y2 = DoubleConv(y2, filters=2*in_out_filters, kernel_size=3, strides=1, batch_norm=batch_norm)

    if squeeze:
        y3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(y2)
        y3 = DoubleConv(y3, filters=4*in_out_filters, kernel_size=3, strides=1, batch_norm=batch_norm)
        y3 = ConvTranspose(y3, filters=2*in_out_filters, conv_transpose=conv_transpose)

        y23 = tf.keras.layers.Concatenate(axis=-1)([y2, y3])
    else:
        y23 = y2

    if double_double_conv:
        y23 = DoubleConv(y23, filters=2*in_out_filters, kernel_size=3, strides=1, batch_norm=batch_norm)

    y23 = ConvTranspose(y23, filters=in_out_filters, conv_transpose=conv_transpose)

    y = tf.keras.layers.Concatenate(axis=-1)([y1, y23])

    y = DoubleConv(y, filters=in_out_filters, kernel_size=3, strides=1, batch_norm=batch_norm)
    y = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)

    model = tf.keras.Model(inputs=x, outputs=y)

    return model
