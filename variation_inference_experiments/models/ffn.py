import tensorflow as tf
from tensorflow.keras.layers import Input, Model
import tensorflow_probability as tfp


def create_flipout_ffn(hidden_layers, input_shape, output_shape, tfp.layers.DenseFlipout, activation):

    inputs = Input(shape=input_shape)
    x = inputs
    for units in hidden_layers:
        x = tfp.layers.DenseFlipout(units=units, activation='relu', padding='same')(x)
    outputs = tfp.layers.DenseFlipout(output_shape, activation=None)(x)

    model = Model(inputs, outputs)

    model.summary()
    return


def create_lrt_ffn(hidden_layers, input_shape, output_shape, activation):
    inputs = Input(shape=input_shape)
    x = inputs
    for units in hidden_layers:
        x = tfp.layers.DenseLocalReparameterization(units=units, activation='relu', padding='same')(x)
    outputs = tfp.layers.DenseLocalReparameterization(output_shape, activation=None)(x)

    model = Model(inputs, outputs)

    model.summary()
    return