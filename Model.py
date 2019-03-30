import numpy as np
import tensorflow as tf
import tensorflow.contrib.summary as tf_summary

import tfUtils as Utils
import Policies
import Losses


# TODO: add docstring to model methods
# TODO: add default arguments if some aren't passed


class Model:
    """
    Model is used as NN approximation for Q values

    args:
    dict conf:
        dict eps:
            max: Float <0.;1.> - Starting epsilon used for exploration strategy
            min: Float <0.;1.> - Epsilon towards which we decay over time
            decay_steps - Positive integer - Number of steps after which the eps value hits min value

        q_func_builder: Function which takes input tensor and builds Q_network

        action_size: Positive Integer - How many different actions are possible
        state_size: Positive Integer - Shape of the state


        memory_type: String {"RankBased","Uniform"} - Memory type either RankBased or Uniform
        memory_size: Positive integer - Number of samples kept by Memory, important for RankBased memory
        beta_fn: function -  Returns important constants for RankBased memory

        Optimizer: Tf.optimizer - Optimizer to use for learning
        discount: Float - Used to introduce uncertainty of future and cut back on crazy infinite predictions

        net_update_frequency: Positive integer - Number of steps after which target NN is updated
        net_update_size: Positive float - How much is target NN updated towards learning NN

        double_Q:  Boolean - Whether to use double_Q variant of DQN

        delta_clip: Float - Used for huber loss to avoid dangerously huge Losses
        grad_norm_clip: Float - Number by which we grad_norm_clip gradients

        problem_type: String - Problem category in OpenAI gym
        reload: Boolean - Whether to build new graph or reload old one

        logdir: String - Path to the place where we want to save/from where we want to load learning statistics and
                important info
        sess: tf.Session - To use graph

    """

    def __init__(self, **conf):
        self.action_size = conf["action_size"]
        self.state_size = conf["state_size"]
        self.problem_type = conf["problem_type"]
        self.q_func_builder = conf["q_func_builder"]

        self.net_update_size = conf["net_update_size"] if "net_update_size" in conf.keys() else 1
        self.net_update_frequency = conf["net_update_frequency"] if\
            "net_update_frequency" in conf.keys() else 10000

        self.discount = conf["discount"] if "discount" in conf.keys() else 0.99
        self.delta_clip = conf["delta_clip"] if "delta_clip" in conf.keys() else 0
        self.grad_norm_clip = conf["grad_norm_clip"] if "grad_norm_clip" in conf.keys() else 10

        self.optimizer = conf["optimizer"]

        self.double_Q = conf["double_Q"] if "double_Q" in conf.keys() else True

        self.logdir = conf["logdir"] + "/chekpoints"
        self.sess = conf["sess"]

        self.memory_type = conf["memory_type"]
        if self.memory_type == "RankBased":
            self.N = conf["memory_size"]
            self.beta_fn = conf["beta_fn"]

        conf_for_build = conf["eps"]
        conf_for_build.update({"action_size": self.action_size,})
        conf_for_build.update({})

        self.build_networks(conf["eps"])
        self.build_agent_vars_log()
        self.initialize(conf["reload"])

    def initialize(self, reload):
        """
        Function which reloads or initializes new graph
        reload: Boolean - Whether to reload pretrained graph variable values

        """
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1,
                                    max_to_keep=10,
                                    )

        if reload:
            print("Reloading graph...")
            print("checkpoints")
            ckpt = tf.train.get_checkpoint_state(self.logdir)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            tf_summary.initialize(graph=tf.get_default_graph())

        else:
            print("Initializing the graph...")
            init = tf.global_variables_initializer()
            self.sess.run(init)
            tf_summary.initialize(graph=tf.get_default_graph())

            self.sess.run(self.copy_ops, feed_dict={self.tau: 1.0})

    def build_agent_vars_log(self):
        """Creates tensorflow log for Agent values"""
        with tf.variable_scope("agent_variable_summaries"):
            self.episode_reward = tf.Variable(0.0,
                                              "episode_reward",
                                              dtype=tf.float32,
                                              )
            total_values = 20
            every_x_step = total_values * 1000
            factors = np.array([1, 5, 50])

            mean_rewards_and_update_ops = np.array(
                [Utils.moving_mean(self.episode_reward, total_values * factor) for factor in factors])

            self.mean_reward_per_ep, self.update_episode_reward_ops = list(mean_rewards_and_update_ops[:, 0]), list(
                mean_rewards_and_update_ops[:, 1])

            self.play_rew = tf.Variable(initial_value=0,
                                        trainable=False,
                                        name="play_rew",
                                        dtype=tf.float32,
                                        )

            with tf_summary.record_summaries_every_n_global_steps(20000, global_step=self.global_step):
                tf_summary.scalar("play_reward", self.play_rew)

            with tf_summary.record_summaries_every_n_global_steps(every_x_step, global_step=self.global_step):
                for idx, ep_amount in enumerate(factors * total_values):
                    tf_summary.scalar("mean_episode_reward_" + str(ep_amount), self.mean_reward_per_ep[idx])

    def build_networks(self, conf):
        """
        Builds the whole graph
        args: conf - dictionary of epsilon policy strategy
            min: Minimum epsilon to be used
            max: Maximum epsilon to be used
            decay_steps: Over how many steps should epsilon be decayed
            actions_size: Number of possible actions
        """

        self.global_step = tf.train.get_or_create_global_step()
        self.placeholders()
        self.build_target_net()
        self.build_train_net()
        self.build_policy(conf)
        self.build_train_ops()
        self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="training")
        self.target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target")
        self.copy_ops, self.tau = Utils.make_move_towards_ops(self.train_vars, self.target_vars)
        tf.assign(self.tau, self.net_update_size)

    def placeholders(self):
        """
        Buildsplaceholders
        """

        if self.problem_type == 'atari':
            obs_shape = (None, 84, 84, 4)
        else:
            obs_shape = (None, self.state_size)
        with tf.variable_scope("placeholders/learn"):
            self.ph_observations = tf.placeholder(dtype=tf.float32,
                                                  shape=obs_shape,
                                                  name="ph_observations",
                                                  )
            self.ph_next_observations = tf.placeholder(dtype=tf.float32,
                                                       shape=obs_shape,
                                                       name="ph_next_observations",
                                                       )

            self.ph_selected_actions_indices = tf.placeholder(dtype=tf.int32,
                                                              shape=(None, ),
                                                              name="ph_selected_actions_actions",
                                                              )

            self.ph_rewards = tf.placeholder(dtype=tf.float32,
                                             shape=(None, ),
                                             name="ph_rewards",
                                             )

            self.ph_is_done = tf.placeholder(dtype=tf.float32,
                                             shape=(None, ),
                                             name="ph_is_done",
                                             )

    def build_target_net(self):
        """
        Builds target net
        """
        name = "target"
        with tf.variable_scope(name):
            self.Qs_target_net = self.q_func_builder(self.ph_next_observations,
                                                     self.action_size,
                                                     scope="target_net",
                                                     reuse=False,
                                                     )

    def build_train_net(self):
        """
        Builds train net
        """
        name = "training"
        with tf.variable_scope(name):
            self.Qs_train_net = self.q_func_builder(self.ph_observations,
                                                    self.action_size,
                                                    scope="train_net",
                                                    reuse=False,
                                                    )

            if self.double_Q:
                self.Qs_train_net_next = self.q_func_builder(self.ph_next_observations,
                                                             self.action_size,
                                                             scope="train_net",
                                                             reuse=True,
                                                             )

    def build_policy(self, conf):
        """
        Builds policy
        args: conf - dictionary of epsilon policy strategy
            min: Minimum epsilon to be used
            max: Maximum epsilon to be used
            decay_steps: Over how many steps should epsilon be decayed
            actions_size: Number of possible actions
        """
        with tf.variable_scope("action_handles"):
            self.exp_action, self.greedy_action = Policies.eps_greedy(self.Qs_train_net, conf)

    def compute_targets(self):
        """
        Builds ops to calculate TD targets
        """
        with tf.variable_scope("computing_targets"):
            if self.double_Q:
                indices_next_actions = tf.argmax(input=self.Qs_train_net_next,
                                                 axis=1,
                                                 output_type=tf.int32,
                                                 )
                one_hot_next_actions = tf.one_hot(indices=indices_next_actions,
                                                  depth=self.action_size,
                                                  axis=-1,
                                                  dtype=tf.float32,
                                                  )

                next_Qs_inp = self.Qs_target_net * one_hot_next_actions
                next_Qs = tf.reduce_sum(input_tensor=next_Qs_inp,
                                        axis=1,
                                        )

            else:
                next_Qs = tf.reduce_max(input_tensor=self.Qs_target_net,
                                        axis=1,
                                        )

            next_Qs = next_Qs * (1 - self.ph_is_done)

            targets = self.ph_rewards + self.discount * next_Qs
            return targets

    def compute_td_error(self, targets):
        """
        Builds opt to compute TD error
        targets: targets of TD error
        """
        with tf.variable_scope("computing_td_error"):
            actions_one_hot = tf.one_hot(indices=self.ph_selected_actions_indices,
                                         depth=self.action_size,
                                         dtype=tf.float32,
                                         )

            responsible_outs = actions_one_hot * self.Qs_train_net
            responsible_outs = tf.reduce_sum(input_tensor=responsible_outs,
                                             axis=1
                                             )

            self.td_error = tf.stop_gradient(targets) - responsible_outs
            if self.memory_type == "RankBased":
                with tf.variable_scope("RankBased_weighting"):
                    self.probs = tf.placeholder(dtype=tf.float32,
                                                shape=(None,),
                                                name="probabilities",
                                                )

                    self.consts = tf.placeholder(dtype=tf.float32,
                                                 shape=(2,),
                                                 name="beta_weight_constants"
                                                 )

                    probs_to_beta = tf.pow(self.probs, self.consts[0])

                    normalize_const_to_beta = tf.pow(self.consts[1], self.consts[0])

                    self.weights = normalize_const_to_beta / probs_to_beta

                    error = self.td_error * self.weights
            else:
                error = self.td_error

            return error

    def build_train_ops(self):
        """
        Builds ops to optimize the graph
        """
        with tf.variable_scope("train_ops"):
            targets = self.compute_targets()
            error = self.compute_td_error(targets)
            with tf.variable_scope("loss"):
                if self.delta_clip > 0:
                    loss = Losses.huber_loss(error, self.delta_clip)
                else:
                    loss = Losses.MSE(error)

            with tf.variable_scope("optimize"):
                grads_and_vars = self.optimizer.compute_gradients(loss,
                                                                  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                                    scope="training",))

                capped_grads_and_vars = [(tf.clip_by_norm(grad, self.grad_norm_clip), var)
                                         for grad, var in grads_and_vars]

                self.train_step = self.optimizer.apply_gradients(grads_and_vars=capped_grads_and_vars,
                                                                 global_step=self.global_step,
                                                                 name="trainStep",
                                                                 )

        with tf.variable_scope("summaries"):
            with tf_summary.record_summaries_every_n_global_steps(2000, global_step=self.global_step):
                tf_summary.scalar("td_error", tf.reduce_mean(self.td_error, axis=0))
                tf_summary.scalar("loss", loss)
                if self.memory_type == "RankBased":
                    tf_summary.scalar("beta", self.consts[0])

            with tf_summary.record_summaries_every_n_global_steps(10000, global_step=self.global_step):
                tf_summary.histogram("td_error", self.td_error)
                if self.memory_type == "RankBased":
                    tf_summary.histogram("weights", self.weights)
                    tf_summary.histogram("probabilities", self.probs)

    def run_copy_ops(self):
        self.sess.run(self.copy_ops, feed_dict={self.tau: self.net_update_size})

    def save(self):
        self.saver.save(self.sess, self.logdir + "/my-model", global_step=self.global_step)

    def update_mean_ep_reward(self, new_reward):
        self.sess.run(self.update_episode_reward_ops, feed_dict={self.episode_reward: new_reward})

    def get_mean_reward_per_ep_1000(self):
        return self.sess.run(self.mean_reward_per_ep[-1])

    def get_mean_reward_per_ep_100(self):
        return self.sess.run(self.mean_reward_per_ep[-2])

    def update_play_rew(self, new_reward):
        self.sess.run(tf.assign(self.play_rew, new_reward))

    def get_glob_step(self):
        return self.sess.run(self.global_step)

    def predict(self, observation):
        feed_dict = {self.ph_observations: observation}
        Q_s = self.sess.run(self.Qs_train_net, feed_dict=feed_dict)
        return Q_s

    def predict_one(self, single_observation):
        feed_dict = {self.ph_observations: single_observation[None]}
        Q_s = self.sess.run(self.Qs_train_net, feed_dict=feed_dict)
        return Q_s[0]

    def act_epsilon(self, single_observation):
        feed_dict = {self.ph_observations: single_observation[None]}
        return self.sess.run(self.exp_action, feed_dict=feed_dict)

    def act(self, single_observation):
        feed_dict = {self.ph_observations: single_observation[None]}
        return self.sess.run(self.greedy_action, feed_dict=feed_dict)

    def train(self, observations, chosen_actions, rewards, next_observations, is_done, probs=None):
        """
        Takes data and performs NNs optimization step. Updates target net or saves model if needed
        """

        if self.memory_type == "RankBased":
            consts = self.beta_fn()
            feed_dict = {self.ph_observations: observations,
                         self.ph_next_observations: next_observations,
                         self.ph_selected_actions_indices: chosen_actions,
                         self.ph_rewards: rewards,
                         self.ph_is_done: is_done,
                         self.probs: probs,
                         self.consts: consts,
                         }
        else:
            feed_dict = {self.ph_observations: observations,
                         self.ph_next_observations: next_observations,
                         self.ph_selected_actions_indices: chosen_actions,
                         self.ph_rewards: rewards,
                         self.ph_is_done: is_done,
                         }

        handles_to_run = [self.td_error, self.global_step, self.train_step, tf_summary.all_summary_ops()]

        td_error, glob_step = self.sess.run(handles_to_run,
                                            feed_dict=feed_dict)[:2]

        if glob_step % self.net_update_frequency == 0:
            self.run_copy_ops()

        if glob_step % 100000 == 0:
            self.save()

        return td_error
