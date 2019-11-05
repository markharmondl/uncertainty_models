from sklearn.model_selection import train_test_split
import tensorflow as tf


def load():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    return {'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test}


