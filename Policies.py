import tensorflow as tf
import tensorflow.contrib.summary as tf_summary


def eps_greedy(Q_layer, conf):
    """
    Takes configuration of Eps policy and Deep Q net to create action handles
    Q_layer: Handle to deep Q net output
    conf: Dictionary of epsilon policy strategy
        min: Minimum epsilon to be used
        max: Maximum epsilon to be used
        decay_steps: Over how many steps should epsilon be decayed
        actions_size: Number of possible actions
    returns:
        handles to: action, epsilon action
    """

    with tf.variable_scope("Actions"):
        batch_size = tf.shape(Q_layer)[0]

        with tf.variable_scope("epsilon_computation"):
            eps_min = tf.Variable(initial_value=conf["min"],
                                  trainable=False,
                                  dtype=tf.float32,
                                  name="end_epsilon",
                                  )
            eps_max = tf.Variable(initial_value=conf["max"],
                                  trainable=False,
                                  dtype=tf.float32,
                                  name="start_epsilon",
                                  )
            decay_steps = tf.Variable(initial_value=conf["decay_steps"],
                                      trainable=False,
                                      dtype=tf.float32,
                                      name="decay_steps",
                                      )
            decay_ratio = tf.cast(tf.train.get_global_step(), tf.float32) / decay_steps
            decay_ratio = tf.minimum(decay_ratio, 1)
            factor = 1 - decay_ratio
            curr_eps = (eps_max - eps_min) * factor + eps_min

        epsilons = tf.random_uniform(shape=(batch_size,),
                                     dtype=tf.float32,
                                     )

        max_ind = tf.argmax(Q_layer, axis=1, output_type=tf.int32)

        rand_actions = tf.random_uniform(shape=(batch_size,),
                                         maxval=conf["action_size"],
                                         dtype=tf.int32,
                                         )

        mask = tf.less(epsilons, curr_eps)
        actions = tf.where(mask,
                           x=rand_actions,
                           y=max_ind,
                           name="greedy_actions_mask",
                           )

        actions = tf.squeeze(actions)
        greedy_actions = tf.squeeze(max_ind)

        with tf_summary.record_summaries_every_n_global_steps(50000, global_step=tf.train.get_global_step()):
            tf_summary.scalar("eps", curr_eps)

        return actions, greedy_actions
