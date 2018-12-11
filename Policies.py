import tensorflow as tf
import tensorflow.contrib.summary as tf_summary

def eps_greedy(Q_layer,conf):
    
    with tf.variable_scope("Actions"):
        batch_size = tf.shape(Q_layer)[0]
        curr_eps = tf.train.polynomial_decay(learning_rate = float(conf["max"]),
                                             global_step = tf.train.get_global_step(),
                                             end_learning_rate = conf["min"],
                                             decay_steps = conf["decay_steps"],
                                             name = "epsilon_computation",
                                             )
        epsilons = tf.random_uniform(shape = (batch_size,),
                                     dtype = tf.float32,
                                     )
        
        max_ind = tf.argmax(Q_layer, axis = 1)
        rand_actions = tf.random_uniform(shape = (batch_size,),
                                         maxval = conf["action_size"],
                                         dtype = tf.int64,
                                         )
        
        
        
        
        mask = tf.less(epsilons,curr_eps)
        actions = tf.where(mask,
                           x=rand_actions,
                           y=max_ind,
                           name="greedy_actions_mask",
                           )
        
    
        actions = tf.squeeze(actions)
        greedy_actions = tf.squeeze(max_ind)
        
        
        
        with tf_summary.record_summaries_every_n_global_steps(50,global_step = tf.train.get_global_step()):
            tf_summary.scalar("eps",curr_eps)

        return (actions,greedy_actions)