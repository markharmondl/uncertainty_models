import tensorflow as tf
from tensorflow.keras.layers import Input, Model
import tensorflow_probability as tfp


def create_flipout_ffn(hidden_layers, input_shape, output_shape, activation):

    inputs = Input(shape=input_shape)
    x = inputs
    for units in hidden_layers:
        x = tfp.layers.DenseFlipout(units=units, activation='relu', padding='same')(x)
    outputs = tfp.layers.DenseFlipout(output_shape, activation=None)(x)

    model = Model(inputs, outputs)

    model.summary()
    return


def create_lrt_ffn(hidden_layers, input_shape, output_shape, activation):
    inputs = Input(input_shape)

    neural_net = tf.keras.Sequential([
        tfp.layers.Convolution2DFlipout(6,
                                        kernel_size=5,
                                        padding="SAME",
                                        activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                     strides=[2, 2],
                                     padding="SAME"),
        tfp.layers.Convolution2DFlipout(16,
                                        kernel_size=5,
                                        padding="SAME",
                                        activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                     strides=[2, 2],
                                     padding="SAME"),
        tfp.layers.Convolution2DFlipout(120,
                                        kernel_size=5,
                                        padding="SAME",
                                        activation=tf.nn.relu),
        tf.keras.layers.Flatten(),
        tfp.layers.DenseFlipout(84, activation=tf.nn.relu),
        tfp.layers.DenseFlipout(10)
    ])

    return