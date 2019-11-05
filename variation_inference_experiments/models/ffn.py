import tensorflow as tf
from tensorflow.keras.layers import Input, Model
import tensorflow_probability as tfp


def create_flipout_ffn(hidden_layers, input_shape, output_shape, activation):

    inputs = Input(shape=input_shape)
    network_input = inputs
    for units in hidden_layers:
        network_input = tfp.layers.DenseFlipout(units=units, activation=activation)(network_input)
    outputs = tfp.layers.DenseFlipout(output_shape, activation=None)(network_input)

    model = Model(inputs, outputs)

    data_x = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
    data_y = tf.placeholder(shape=[None], dtype=tf.int32)
    penalty = tf.placeholder(shape=[], dtype=tf.float32)

    logits = model(data_x)
    probs = tf.nn.softmax(logits, axis=1)
    labels_distribution = tfp.distributions.Categorical(logits=logits)
    log_probs = labels_distribution.log_prob(data_y)

    neg_log_likelihood = -tf.reduce_mean(log_probs)
    kl = sum(model.losses) / penalty
    elbo_loss = neg_log_likelihood + kl

    correct_preds = tf.equal(tf.cast(data_y, dtype=tf.int64), tf.argmax(probs, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

    return {'model': model,
            'accuracy': accuracy,
            'elbo_loss': elbo_loss,
            'probs': probs,
            'neg_log_liklihood': neg_log_likelihood}
