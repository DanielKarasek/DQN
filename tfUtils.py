import numpy as np
import tensorflow as tf


def moving_mean(new_value, total_len, name="moving_mean"):
    """
    Function building handle to mean of last n given values and handle to
    add these new values.
    Args:
        new_value: tf.Variable - Handle to new variable fed into model
        total_len: int - Decides how many values are kept
    Kwargs:
        scope: String - Name of variable scope in which this part of graph will be wrapped

    Returns:
        mean: tf.Variable - Handle to variable representing current mean value
        update_ops: tf.Operation - Handle to operations which add new value and
                                   and shreds old one

    """

    with tf.variable_scope(name):
        current_pos = tf.Variable(initial_value=0,
                                  trainable=False,
                                  name="current_position_in_mem",
                                  dtype=tf.int64,
                                  )

        element_tensor = tf.Variable(initial_value=np.zeros(total_len,
                                                            dtype=np.float64,
                                                            ),
                                     trainable=False,
                                     name="mean_memory",
                                     )

        op1 = element_tensor[current_pos].assign(new_value)
        op2 = current_pos.assign((current_pos + 1) % total_len)
        update_ops = [op1, op2]

        mean = tf.reduce_sum(element_tensor) / total_len

    return mean, update_ops


def make_move_towards_ops(src_vars, dest_vars, scope="moving_target"):
    """
    Function to build ops for 2 set of variables. These ops move values of
    one set closer to the other by value tau, which is fed through feed_dict.
    Args:
        src_vars: tf.Variables - Handle to weights towards which move dest_vars
        dest_vars: tf.Variables - Handle to weights to move towards src_vars
    Kwargs:
        scope: String - Name of variable scope in which this part of graph will be wrapped

    Returns:
        ops: tf.Operations - Handles to operations.
        tau: tf.Variable - Handle to Variable tau, so any value can be fed as tau
    """

    with tf.variable_scope(scope):
        ops = []

        tau = tf.Variable(initial_value=1.0,
                          trainable=False,
                          name="tau",
                          dtype=tf.float64,
                          )

        for src_var, dest_var in zip(src_vars, dest_vars):
            ops.append(dest_var.assign(tf.add(tf.multiply(tau, src_var),
                                              tf.multiply(1 - tau, dest_var))))
        return ops, tau
