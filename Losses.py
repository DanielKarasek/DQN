import tensorflow as tf


"""
This module contains basic Losses used for NNs
"""


def huber_loss(x, delta_clip):
    loss = tf.where(condition=tf.abs(x) < delta_clip,
                    x=tf.square(x) * 0.5,
                    y=delta_clip * (tf.abs(x) - 0.5 * delta_clip)
                    )
    return tf.reduce_mean(loss)


def MSE(td_errors):
    squaredDiff = 0.5 * tf.square(td_errors)
    loss = tf.reduce_mean(input_tensor=squaredDiff,
                          axis=0,
                          )                           
    return loss


def MAE(td_errors):
    return tf.reduce_mean(input_tensor=tf.abs(td_errors),
                          axis=0,
                          )
