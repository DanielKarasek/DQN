import numpy as np
import tensorflow as tf
import tensorflow.contrib.summary as tf_summary

import tfUtils as Utils
import Policies
import Losses
import Nets


# noinspection PyAttributeOutsideInit
class Model:

    def __init__(self, **conf):
        self.double_Q = conf["double_Q"]

        self.net_update_size = conf["net_update_size"]
        self.net_update_frequency = conf["net_update_frequency"]

        self.logdir = conf["logdir"] + "/chekpoints"
        self.sess = conf["sess"]

        self.memory_type = conf["memory_type"]
        if self.memory_type == "RankBased":
            self.N = conf["memory_size"]
            self.beta_fn = conf["beta_fn"]

        self.action_size = conf["action_size"]
        self.state_size = conf["state_size"]

        self.discount = conf["discount"]

        self.optimizer = self.create_optimizer(conf["lr"])

        self.delta_clip = conf["delta_clip"]

        conf_for_build = conf["eps"]
        conf_for_build.update({"action_size": self.action_size})
        conf_for_build.update({"layers": conf["layers"]})

        self.build_networks(conf["eps"])
        self.build_agent_vars_log()
        self.initialize(conf["reload"])

    def initialize(self, reload):
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1,
                                    max_to_keep=10,
                                    )

        if reload:
            print("Reloading graph...")
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
        with tf.variable_scope("agent_variable_summaries"):
            self.episode_reward = tf.Variable(0.0,
                                              "episode_reward",
                                              dtype=tf.float64,
                                              )
            total_values = 20
            every_x_step = total_values * 1000
            factors = np.array([1, 5, 50])

            mean_rewards_and_update_ops = np.array(
                [Utils.moving_mean(self.episode_reward, total_values * factor) for factor in factors])

            self.mean_reward_per_ep, self.update_episode_reward_ops = list(mean_rewards_and_update_ops[:, 0]), list(
                mean_rewards_and_update_ops[:, 1])

            with tf_summary.record_summaries_every_n_global_steps(every_x_step, global_step=self.global_step):
                for idx, ep_amount in enumerate(factors * total_values):
                    tf_summary.scalar("mean_episode_reward_" + str(ep_amount), self.mean_reward_per_ep[idx])

    def build_networks(self, conf):
        self.global_step = tf.train.get_or_create_global_step()
        self.placeholders()
        self.build_target_net(conf["layers"])
        self.build_train_net(conf["layers"], conf)
        self.build_train_ops()
        self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="training")
        self.target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target")
        self.copy_ops, self.tau = Utils.make_move_towards_ops(self.train_vars, self.target_vars)

    def placeholders(self):
        with tf.variable_scope("placeholders/learn"):
            self.observations = tf.placeholder(dtype=tf.float64,
                                               shape=(None, self.state_size),
                                               name="observations",
                                               )

            self.next_observations = tf.placeholder(dtype=tf.float64,
                                                    shape=(None, self.state_size),
                                                    name="next_observations",
                                                    )

            self.selected_actions_indices = tf.placeholder(dtype=tf.int64,
                                                           shape=None,
                                                           name="selected_actions_actions",
                                                           )

            self.rewards = tf.placeholder(dtype=tf.float64,
                                          shape=None,
                                          name="rewards",
                                          )

            self.is_done_mask = tf.placeholder(dtype=tf.bool,
                                               shape=None,
                                               name="is_done_mask",
                                               )

    def build_target_net(self, layers):
        name = "target"
        with tf.variable_scope(name):
            self.Qs_target_net = Nets.build_dnn(self.next_observations, self.action_size, layers)

    def build_train_net(self, layers, conf):
        name = "training"
        with tf.variable_scope(name) as scope:
            self.Qs_train_net = Nets.build_dnn(self.observations, self.action_size, layers)
            if self.double_Q:
                scope.reuse_variables()
                self.Qs_train_net_next = Nets.build_dnn(self.next_observations, self.action_size, layers)

        with tf.variable_scope("action_handles") as scope:
            self.exp_action, self.greedy_action = Policies.eps_greedy(self.Qs_train_net, conf)

    def compute_targets(self):
        with tf.variable_scope("computing_targets"):
            if self.double_Q:
                indices_next_actions = tf.argmax(input=self.Qs_train_net_next,
                                                 axis=1,
                                                 output_type=tf.int64,
                                                 )
                one_hot_next_actions = tf.one_hot(indices=indices_next_actions,
                                                  depth=self.action_size,
                                                  axis=-1,
                                                  dtype=tf.float64,
                                                  )

                next_Qs_inp = self.Qs_target_net * one_hot_next_actions
                next_Qs = tf.reduce_sum(input_tensor=next_Qs_inp,
                                        axis=1,
                                        )

            else:
                next_Qs = tf.reduce_max(input_tensor=self.Qs_target_net,
                                        axis=1,
                                        )

            next_Qs = tf.where(self.is_done_mask,
                               tf.zeros(shape=tf.shape(next_Qs)[0],
                                        dtype=tf.float64),
                               next_Qs,
                               )
            targets = self.rewards + self.discount * next_Qs

            #             targets = tf.layers.batch_normalization(inputs=tf.reshape(targets,shape=(-1,1)),
            #                                                     fused = True,
            #                                                     training=False,
            #                                                     trainable=False
            #                                                     )

            return tf.reshape(targets, (-1,))

    def compute_td_error(self, targets):
        with tf.variable_scope("computing_td_error"):
            actions_one_hot = tf.one_hot(indices=self.selected_actions_indices,
                                         depth=self.action_size,
                                         axis=-1,
                                         dtype=tf.float64,
                                         )
            responsible_outs = actions_one_hot * self.Qs_train_net
            responsible_outs = tf.reduce_sum(input_tensor=responsible_outs,
                                             axis=1
                                             )
            self.td_error = tf.abs(responsible_outs - targets)

            if self.memory_type == "RankBased":
                with tf.variable_scope("RankBased_weighting"):
                    self.probs = tf.placeholder(dtype=tf.float64,
                                                shape=(None,),
                                                name="probabilities",
                                                )

                    self.consts = tf.placeholder(dtype=tf.float64,
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
                                                                                    "training"))

                capped_grads_and_vars = [(tf.clip_by_value(grad, -0.5, 0.5), var) for grad, var in grads_and_vars]

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

    def create_optimizer(self, lr):
        RMS = tf.train.RMSPropOptimizer(learning_rate=lr, )
        return RMS

    def run_copy_ops(self):
        self.sess.run(self.copy_ops, feed_dict={self.tau: self.net_update_size})

    def save(self):
        self.saver.save(self.sess, self.logdir + "/my-model", global_step=self.global_step)

    def update_mean_ep_reward(self, new_reward):
        self.sess.run(self.update_episode_reward_ops, feed_dict={self.episode_reward: new_reward})

    def get_mean_reward_per_ep(self):
        return self.sess.run(self.mean_reward_per_ep[-1])

    def get_glob_step(self):
        return self.sess.run(self.global_step)

    def predict(self, observations):
        feed_dict = {self.inputs: observations}
        Q_s = self.sess.run(self.Qs_train_net, feed_dict=feed_dict)
        return Q_s

    def predict_one(self, single_observation):
        feed_dict = {self.observations: [single_observation]}
        Q_s = self.sess.run(self.Qs_train_net, feed_dict=feed_dict)
        return Q_s[0]

    def act_epsilon(self, observations):
        feed_dict = {self.observations: [observations]}
        return self.sess.run(self.exp_action, feed_dict=feed_dict)

    def act(self, observations):
        feed_dict = {self.observations: [observations]}
        return self.sess.run(self.greedy_action, feed_dict=feed_dict)

    def train(self, observations, chosen_actions, rewards, next_observations, is_done_mask, probs=None):
        if self.memory_type == "RankBased":
            consts = self.beta_fn()
            feed_dict = {self.observations: observations,
                         self.next_observations: next_observations,
                         self.selected_actions_indices: chosen_actions,
                         self.rewards: rewards,
                         self.is_done_mask: is_done_mask,
                         self.probs: probs,
                         self.consts: consts,
                         }
        else:
            feed_dict = {self.observations: observations,
                         self.next_observations: next_observations,
                         self.selected_actions_indices: chosen_actions,
                         self.rewards: rewards,
                         self.is_done_mask: is_done_mask,
                         }

        handles_to_run = [self.td_error, self.global_step, self.train_step, tf_summary.all_summary_ops()]

        td_error, glob_step = self.sess.run(handles_to_run,
                                            feed_dict=feed_dict)[:2]

        #debug code
        # handles_to_run = [self.td_error, self.train_step]
        # td = self.sess.run(handles_to_run, feed_dict=feed_dict)[0]
        # return td
        #/debug code

        if self.double_Q:
            if glob_step % self.net_update_frequency == 0:
                self.run_copy_ops()

        # if glob_step % 100000 == 0:
        #     self.save()

        # create generator ?
        return td_error
