import tensorflow as tf


def load_and_shuffle(batch_size, seed):
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    train_data_tf = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(seed).batch(batch_size)

    test_data_tf = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    return {'train_data_tf': train_data_tf,
            'test_data_tf': test_data_tf}


