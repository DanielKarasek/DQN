from copy import deepcopy

import pickle
import os
import time

import numpy as np
import tensorflow.contrib.summary as tf_summary

from Memory import *
from Model import Model
from Utils import my_logger


class Agent():
    '''
    Agent used to solve problems via DQN/DDQN
    
    args:
    dict conf: 
        dict memory:
            size: Positive integer - Number of samples kept by memory
            type: String {"RankBased","Uniform"} - Memory type either RankBased or Uniform
            batch_size: Positive integer - Batch size used for learning
            alpha: Float <0.;1.> - Alpha, used only in case of memory type RankBased
            beta: Float <0.;1.> - Starting beta, used only in case of memory type RankBased
            total_beta_time: Positive integer - Number of steps to ascend beta towards 1, used only in case of memory type RankBased
        dict hyper_params:
            lr: Float <0.;1.> - Learning Rate used for NN optimization
            net_update_size: Positive integer - Number of steps after which target net is updated
            discount: Float <0.;1.> - discount used in target equation
            double_Q: Boolean - use double_Q variant of DQN
        dict eps:
            max: Float <0.;1.> - Starting epsilon used for exploration strategy 
            min: Float <0.;1.> - Epsilon towards which we decay over time 
            decay_steps - Positive integer - Number of steps after which the eps value hits min value
        logdir: String - Relative path to place to save learning statistics
        env: OPEN AI env - One of OPEN AI environments, use only the 3 suggested, others might not be supported ["CartPole-v1", "LunarLander-v2", "MountainCar-v0"]
        sess: tf.session - To use graph
    '''

    @my_logger
    def __init__(self, conf):
        print("Initializing Agent...")
        self.env = conf["env"]

        self.logdir = conf["logdir"]
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape[0]

        self.sess = conf["sess"]

        if conf["reload"]:
            with open(self.logdir + "/agent.txt", "rb") as myFile:
                conf = pickle.load(myFile)
            conf.update({"sess": self.sess, "env": self.env, "reload": True})
            self.memory_type = conf["memory"]["type"]
            self.memory_size, beta_fn = self.load_memory()
            self.ram = conf["RAM"]

            conf["memory"]["size"] = self.memory_size

            self.memory_type = conf["memory"]["type"]

            logdir_path_abs = os.getcwd() + "/" + conf["logdir"]

            self.summary_writer = tf_summary.create_file_writer(logdir_path_abs)
            self.summary_writer.set_as_default()

            train_dict = self._create_train_dict(conf, beta_fn)

            self.logging_dict = self._init_log_dict()

            print("Building train and target graphs..")
            self.train_model = Model(**train_dict)

        else:
            self.ram = conf["RAM"]

            self.memory_size, beta_fn = self._create_memory(conf["memory"])

            conf["memory"]["size"] = self.memory_size

            self.memory_type = conf["memory"]["type"]

            logdir_path_abs = os.getcwd() + "/" + conf["logdir"]

            self.summary_writer = tf_summary.create_file_writer(logdir_path_abs, name="eventssss")
            self.summary_writer.set_as_default()

            train_dict = self._create_train_dict(conf, beta_fn)

            self.logging_dict = self._init_log_dict()

            print("Building train and target graphs..")
            self.train_model = Model(**train_dict)

            conf.pop("sess")
            conf.pop("env")
            conf.pop("reload")
            with open(self.logdir + "/agent.txt", "wb") as myFile:
                pickle.dump(conf, myFile)

    def _measure_train(self, transition):
        time_start = time.time()
        if self.memory_type == "RankBased":
            samples, probabilities = self.memory.get_n_samples()
            probabilities = np.append(probabilities, self.median_probability)
            self.memory.unset_need_update()
        else:
            samples = self.memory.get_n_samples()

        samples = np.append(samples,
                            transition[:, np.newaxis],
                            axis=1,
                            )

        s, a, r, s_, d = samples
        s = np.vstack(s)
        s_ = np.vstack(s_)

        time_end = time.time()
        timer_preproc = time_end - time_start

        if self.memory_type == "RankBased":
            train_start = time.time()
            for _ in range(100):
                td = self.train_model.train(s, a, r, s_, d, probs=probabilities)

            timer_train_ops = (time.time() - train_start) / 100
            timer_update_mem = time.time()
            self.memory.add_transition(transition, td[-1])
            timer_update_mem = time.time() - timer_update_mem
            timer_total = time.time() - time_start - 99 * timer_train_ops
            return timer_preproc, timer_train_ops, timer_update_mem, timer_total

        else:
            train_start = time.time()
            for _ in range(100):
                self.train_model.train(s, a, r, s_, d)
            timer_train_ops = (time.time() - train_start) / 100
            timer_update_mem = time.time()
            self.memory.add_transition(transition)
            timer_total = time.time() - time_start - 99 * timer_train_ops
            timer_update_mem = time.time() - timer_update_mem
            return timer_preproc, timer_train_ops, timer_update_mem, timer_total

    def _create_train_dict(self, conf, beta_fn=None):
        '''
        creates dict of arguments for model
        '''
        train_dict = {}
        train_dict["reload"] = conf["reload"]
        train_dict["action_size"] = self.action_size
        train_dict["state_size"] = self.state_size
        train_dict["lr"] = conf["hyper_params"]["lr"]
        train_dict["sess"] = conf["sess"]
        train_dict["memory_type"] = conf["memory"]["type"]
        train_dict["memory_size"] = conf["memory"]["size"]
        train_dict["delta_clip"] = 0.5
        train_dict["eps"] = conf["eps"]
        train_dict["discount"] = conf["hyper_params"]["discount"]
        train_dict["logdir"] = conf["logdir"]
        train_dict["double_Q"] = conf["hyper_params"]["double_Q"]
        train_dict["beta_fn"] = beta_fn
        train_dict["net_update_size"] = conf["hyper_params"]["net_update_size"]
        train_dict["net_update_frequency"] = conf["hyper_params"]["net_update_frequency"]
        train_dict["layers"] = conf["hyper_params"]["layers"]

        return train_dict

    def _init_tracked_event(self, tracked_event_name, n_steps, log_dict):
        """
        Adds dictionary of wanted tracked event in log_dict

        returns: None

        args:
            tracked_event_name: basic data type - name of tracked event
            n_steps: Iterable - every object is number of steps after which we wish to track given event
            log_dict: Dictionary - Dictionary in which we wish to add our new tracked event
        """
        try:
            log_dict[tracked_event_name] = {}
            for n_step in n_steps:
                log_dict[tracked_event_name][n_step] = {}
                log_dict[tracked_event_name][n_step]['c'] = 1
                log_dict[tracked_event_name][n_step]['v'] = 0
        except TypeError:
            message = "_init_tracked_event(Agent): object n_steps must be iterable. Occured with event name " + tracked_event_name
            print(message)
            # raise TypeError(message)            <- clean version, ^^ debug version

    def _init_log_dict(self):
        log_dict = {}
        episode_time_space = np.logspace(1, 3, 3, 10)
        self._init_tracked_event("episode_time", episode_time_space, log_dict)
        return log_dict

    def _process_RAM_state(self, state):
        return np.divide((state - 127.5), 127.5,
                         dtype=np.float32)

    def _create_memory(self, memory_conf):
        """
        Creates and fills memory with random experience

        returns: Size of memory,beta weight function if it exists
        (RankBased memory size is only in powers of two,
        if another arg is passed RankBased finds the closest higher power of 2)

        args:
            dict memory:
                size: Positive integer - Number of samples kept by memory
                type: String {"RankBased","Uniform"} - Memory type either RankBased or Uniform
                batch_size: Positive integer - Batch size used for learning
                alpha: Float <0.;1.> - Alpha, used only in case of memory type RankBased
                beta: Float <0.;1.> - Starting beta, used only in case of memory type RankBased
                total_beta_time: Positive integer - Number of steps to ascend beta towards 1, used only in case of memory type RankBased
        """

        print("Creating memory and filling it with random experience...")
        if memory_conf["type"] == "RankBased":
            self.memory = MemoryRankBased(**memory_conf)
            self.median_probability = self.memory.probabilities[int(self.memory.max_size // 2)]
            self._fill_memory(True)
            print("memory filled")
            return self.memory.max_size, lambda: self.memory.get_beta_plus_weight(self.sess)

        else:
            self.memory = MemoryUniform(**memory_conf)
            self._fill_memory()
            print("memory filled")
            return self.memory.max_size, None

    def _fill_memory(self, RANK_FLAG=False):
        while not self.memory.is_full():
            if self.ram:
                s = self._process_RAM_state(self.env.reset())
            else:
                s = self.env.reset()
            done = False
            while not done:
                a = np.random.randint(self.action_size)
                s_, r, done, _ = self.env.step(a)

                s_ = self._process_RAM_state(s_) if self.ram else s_

                transition = np.array([np.array(s, copy=True), a, r, s_, done])
                if RANK_FLAG:
                    self.memory.add_transition(transition, np.random.rand())
                else:
                    self.memory.add_transition(transition)

                s = s_

    def _train(self, transition):
        if self.memory_type == "RankBased":
            samples, probabilities = self.memory.get_n_samples()
            probabilities = np.append(probabilities, self.median_probability)
        else:
            samples = self.memory.get_n_samples()

        samples = np.append(samples,
                            transition[:, np.newaxis],
                            axis=1,
                            )

        s, a, r, s_, d = samples
        s = np.vstack(s)
        s_ = np.vstack(s_)

        if self.memory_type == "RankBased":

            # td = self.train_model.train(s, a, r, s_, d, probs=probabilities)
            # debug code
            td = self.train_model.train(s, a, r, s_, d, probs=probabilities)
            self.memory.update(td[:-1])
            self.memory.add_transition(transition=transition, priority=td[-1])

            # /debug code
            # self.memory.update(td[:-1])
            # self.memory.add_transition(transition, td[-1])

        else:
            self.train_model.train(s, a, r, s_, d)
            self.memory.add_transition(transition)

    @my_logger
    def solve(self,
              flag_container={"done": False, "show": False, "tune": False},
              verbosity=1, ):

        if verbosity > 0:
            print("learning started")

        ep_counter = 0
        step_counter = self.train_model.get_glob_step()
        total_length = 20e6

        while True:
            time_start = time.time()

            steps = 0
            R = 0
            if self.ram:
                s = self._process_RAM_state(self.env.reset())
            else:
                s = self.env.reset()
            done = False
            start_Qs = self.train_model.predict_one(s)

            while not done:
                if flag_container["show"]:
                    self.env.render()
                    time.sleep(0.01)
                if step_counter % self.memory_size == 0 and self.memory_type == "RankBased":
                    print("Memory sorting")
                    self.memory.heap_sort()

                a = self.train_model.act_epsilon(s)

                s_, r, done, _ = self.env.step(a)

                s_ = self._process_RAM_state(s_) if self.ram else s_

                transition = np.array([deepcopy(s), a, r, s_, done])
                self._train(transition)

                steps += 1
                step_counter += 1
                R += r
                s = s_

            ep_counter += 1
            time_end = time.time()

            self.handle_episode_timer(time_start, time_end)
            self.train_model.update_mean_ep_reward(R)

            if flag_container["tune"] and step_counter > total_length:
                self.memory.save(self.logdir)
                return self.train_model.get_mean_reward_per_ep()

            if flag_container["done"]:
                self.memory.save(self.logdir)
                return 0

            if ep_counter % 20 == 0:
                last_Qs = self.train_model.predict_one(s)
                print("##################")
                print("It took ", time_end - time_start, "seconds from last print timeeee ")
                print("Average time per time-step is", (time_end - time_start) / steps)
                print("last Qs: ", last_Qs)
                print("problem start Qs: ", start_Qs)
                print("reward for ", ep_counter, "-th episode is : ", R)
                print("##################")

    def play(self, n_steps=None, flag_container={"done": False, "show": True}):
        ep_counter = 0
        total_reward = 0
        curr_num_steps = 0

        while True:
            R = 0
            s = self._process_RAM_state(self.env.reset()) if self.ram else self.env.reset()
            done = False

            while not done:
                if flag_container["show"]:
                    self.env.render()
                    time.sleep(0.01)

                a = self.train_model.act(s)

                s_, r, done, _ = self.env.step(a)

                s_ = self._process_RAM_state(s_) if self.ram else s_

                curr_num_steps += 1
                R += r
                s = s_

                if n_steps is not None and curr_num_steps == n_steps:
                    return total_reward / ep_counter

                if flag_container["done"]:
                    return total_reward / ep_counter

            total_reward += R
            ep_counter += 1

            print("##################")
            print("reward for ", ep_counter, "-th episode is : ", R)
            print("average reward for all episodes so far: ", total_reward / ep_counter)
            print("##################")

    def constant_time(self, s):

        self.train_model.predict_one(s)
        self.train_model.train([s], [0], [1], [s], [0], [0.7])
        self.train_model.act_epsilon(s)

        time_start = time.time()
        self.train_model.predict_one(s)
        time_end = time.time()
        time_final = (time_end - time_start)
        print("One feed to NN takes ", time_final, "seconds")

        time_start = time.time()
        self.train_model.act_epsilon(s)
        time_end = time.time()
        time_final = (time_end - time_start)
        print("One feed to NN and get eps actions takes ", time_final, "seconds")

        time_start = time.time()
        self.train_model.run_copy_ops()
        time_end = time.time()
        time_final = time_end - time_start
        print("One full_copy step takes ", time_final, "seconds")

        t_preproc, t_train, t_update, t_total = self._measure_train(np.array([s, 0, 1, s, 0, ]))
        print("Preproccing in _train takes ", t_preproc, "seconds")
        print("Calling anc computing train_ops takes ", t_train, "seconds")
        print("One memory update step ", t_update, "seconds")
        print("One train_step takes ", t_total, "seconds")

    def handle_episode_timer(self, time_start, time_end):
        episode_time = time_end - time_start
        dic = self.logging_dict["episode_time"]
        for x in 10, 100, 1000:
            dic[x]['v'] = dic[x]['v'] + (episode_time - dic[x]['v']) / dic[x]['c']
            if dic[x]['c'] == x:
                print("Average time spend in last ", x, " episodes is", dic[x]['v'], "seconds")
                dic[x]['c'] = 0
            dic[x]['c'] += 1

    def load_memory(self):
        conf = {"logdir": self.logdir}
        if self.memory_type == "RankBased":
            self.memory = MemoryRankBased(True, **conf)
            self.median_probability = self.memory.probabilities[int(self.memory.max_size // 2)]
            return self.memory.max_size, lambda: self.memory.get_beta_plus_weight(self.sess)
        else:
            pass
