import tensorflow as tf


def huber_loss(td_errors,delta_clip):
    quadratic_part = tf.clip_by_value(t=td_errors, 
                                      clip_value_min=0, 
                                      clip_value_max=delta_clip
                                      )
    
    linear_part = td_errors - quadratic_part
    loss = 0.5 * tf.square(quadratic_part) + delta_clip * linear_part
    loss = tf.reduce_mean(input_tensor=loss,
                          axis=0,
                          )  
    return loss
    
    
def MSE(td_errors):
    squaredDiff = 0.5 * tf.square(td_errors)
    loss = tf.reduce_mean(input_tensor=squaredDiff,
                          axis=0,
                          )                           
    return loss


def MAE(td_errors):
    return tf.reduce_mean(input_tensor=td_errors,
                          axis=0,
                          )