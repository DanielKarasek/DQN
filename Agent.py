import time

import numpy as np
import cv2
import tensorflow as tf


from Memory import *
from Model import Model
from Utils import my_logger, unpickle, pickle_obj
from tfUtils import create_summary_writer
from Nets import get_network_builder


class Agent(object):
    """
    Agent used to solve problems via DQN/DDQN

    args:
    dict conf:
        dict memory:
            size: Positive integer - Number of samples kept by memory
            batch_size: Positive integer - Batch size used for learning
            type: String {"RankBased","Uniform"} - Memory type either RankBased or Uniform
            alpha: Float <0.;1.> - Alpha, used only in case of memory type RankBased
            beta: Float <0.;1.> - Starting beta, used only in case of memory type RankBased
            total_beta_time: Positive integer - Number of steps to ascend beta towards 1, used only in case of memory
                             type RankBased
        dict hyper_params:
            lr: Float <0.;1.> - Learning Rate used for NN optimization
            discount: Float <0.;1.> - discount used in target equation
            layers: Iterable - Set of positive integers representing size of DNN layers
            net_update_frequency: Positive integer - Number of steps after which target NN is updated
            net_update_size: Positive float - How much is target NN updated towards learning NN
            double_Q: Boolean - use double_Q variant of DQN
            dueling: Boolean - whether to use dueling model of NNs
        dict eps:
            max: Float <0.;1.> - Starting epsilon used for exploration strategy
            min: Float <0.;1.> - Epsilon towards which we decay over time
            decay_steps - Positive integer - Number of steps after which the eps value hits min value
        logdir: String - Path to the place where we want to save/from where we want to load learning statistics and
                important info
        env: OPEN AI env - One of OPEN AI environments (Supported are classic, atari envs)
        env_play: OPEN AI env - One of OPEN AI environments (Supported are classic, atari envs)
        problem_type: String - Type of OPEN AI env
        sess: tf.Session - To use graph
        reload: Boolean - reset learning saved in logdir
    """

    # TODO: add possibility to change mem size etc. when loading

    @my_logger
    def __init__(self, conf):
        print("Initializing Agent...")
        self.env = conf["env"]
        self.env_play = conf["env_play"]

        self.logdir = conf["logdir"]
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape[0]

        self.sess = conf["sess"]

        self.logging_dict = self._init_log_dict()

        if conf["reload"]:
            self._reload_agent()
            conf["reload"] = False
        else:
            self._init_rest(conf)

    def _init_rest(self, conf):
        """
        Initializes rest of the agent when agent isn't reloaded given. This include memory and graph building.

        args:
            dict conf: Dictionary of agents configuration
        """
        self.problem_type = conf["problem_type"]

        self.memory_type = conf["memory"]["type"]
        self.memory_size, beta_fn = self._create_memory(conf["memory"])
        conf["memory"]["size"] = self.memory_size

        print("Building train and target graphs..")

        create_summary_writer(self.logdir)
        train_dict = self._create_train_dict(conf, beta_fn)
        self.train_model = Model(**train_dict)
        self._save_conf(conf)

    def _save_conf(self, conf):
        """
        Saves configuration passed to agent into logdir/agent.txt

        args:
            dict conf: Dictionary of agents configuration
        """
        keys = ["sess", "env", "reload", "env_play"]
        for key in keys:
            conf.pop(key)
        pickle_obj(conf, self.logdir + "/agent.txt")

    def _reload_agent(self):
        """
        Reloads agent from Agents logdir, memory and graph included
        """
        conf = unpickle(self.logdir + "/agent.txt")
        conf.update({"sess": self.sess, "env": self.env, "env_play": self.env_play, "reload": True})
        print("Reloading memory")
        self.problem_type = conf["problem_type"]
        self.memory_type = conf["memory"]["type"]
        self.memory_size, beta_fn = self._load_memory(conf)
        conf["memory"]["size"] = self.memory_size
        print("Memory reloaded")

        print("Building train and target graphs..")
        create_summary_writer(self.logdir)
        train_dict = self._create_train_dict(conf, beta_fn)
        self.train_model = Model(**train_dict)
        print("Graph built")

    def _constant_time(self, s):
        """
        Deprecated
        Measures and prints static times needed for:
            One NN feed to get Q_values
            One NN feed to get eps_action
            One copy ops of NN params
            Statistics of _train via function _measure_train

        args:
            s: np.array - any possible state of environment
        """

        self.train_model.predict_one(s)
        self.train_model.train(s[None], [0], [1], s[None], [0], [0.7])
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

        t_preproc, t_train, t_update, t_total = self._measure_train((s, 0, 1, s, 0, ))
        print("Preproccing in _train takes ", t_preproc, "seconds")
        print("Calling anc computing train_ops takes ", t_train, "seconds")
        print("One memory update step ", t_update, "seconds")
        print("One train_step takes ", t_total, "seconds")

    def _measure_train(self, transition):
        """
        Deprecated
        Function train permuted so we can calculate time requiremnts
        returns: time needed for: Preprocessing, running train_ops, updating memory and total time needed for this function
        """
        time_start = time.time()
        if self.memory_type == "RankBased":
            samples, probabilities = self.memory.get_n_samples()
            probabilities = np.append(probabilities, self.median_probability)
            self.memory.unset_need_update()
        else:
            samples = self.memory.get_n_samples()

        s, a, r, s_, d = samples

        t_s, t_a, t_r, t_s_, t_d = transition
        t_s, t_s_ = np.array(t_s), np.array(t_s_)

        s = np.append(s, t_s[None, ...], axis=0)
        s_ = np.append(s_, t_s_[None, ...], axis=0)
        a = np.append(a, [t_a], axis=0)
        r = np.append(r, [t_r], axis=0)
        d = np.append(d, [t_d], axis=0)


        s = np.stack(s, 0)
        s_ = np.stack(s_, 0)

        time_end = time.time()
        timer_preproc = time_end - time_start

        if self.memory_type == "RankBased":
            train_start = time.time()
            for _ in range(5):
                td = self.train_model.train(s, a, r, s_, d, probs=probabilities)

            timer_train_ops = (time.time() - train_start) / 5
            timer_update_mem = time.time()
            self.memory.add_transition(transition, td[-1])
            timer_update_mem = time.time() - timer_update_mem
            timer_total = time.time() - time_start - 4 * timer_train_ops
            return timer_preproc, timer_train_ops, timer_update_mem, timer_total

        else:
            train_start = time.time()
            for _ in range(100):
                self.train_model.train(s, a, r, s_, d)
            timer_train_ops = (time.time() - train_start) / 100
            timer_update_mem = time.time()
            self.memory.add_transition(transition)
            timer_total = time.time() - time_start - 4 * timer_train_ops
            timer_update_mem = time.time() - timer_update_mem
            return timer_preproc, timer_train_ops, timer_update_mem, timer_total

    def _init_log_dict(self):
        """
        Initializes dictionaries to keep measuring time per 10,100,1000 episodes
        """
        log_dict = {}
        episode_time_space = np.logspace(1, 3, 3, 10)
        self._init_tracked_event("episode_time", episode_time_space, log_dict)
        return log_dict

    @staticmethod
    def _init_tracked_event(tracked_event_name, n_steps, log_dict):
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
            message = "_init_tracked_event(Agent): object n_steps must be iterable. Occured with event name " +\
                      tracked_event_name
            print(message)
            # raise TypeError(message)            <- clean version, ^^ debug version

    def _handle_episode_timer(self, time_start, time_end, verbosity):
        """
        Uses dic to calculate average time of episode for past 10,100,1000 eps

        args:
        time_start: Float - time at which episode started
        time_end: Float - time at which episode ended
        """
        episode_time = time_end - time_start
        dic = self.logging_dict["episode_time"]
        for x in 10, 100, 1000:
            dic[x]['v'] = dic[x]['v'] + (episode_time - dic[x]['v']) / dic[x]['c']
            if dic[x]['c'] == x and verbosity > 0:
                print("Average time spend in last ", x, " episodes is", dic[x]['v'], "seconds")
                dic[x]['c'] = 0
            dic[x]['c'] += 1

    def _create_train_dict(self, conf, beta_fn=None):
        """
        creates dict of arguments for model
        """
        batch_norm = True
        train_dict = {"q_func_builder": get_network_builder("deepQ")(net_name="conv2fully",
                                                                     **{"batch_norm": batch_norm,
                                                                        "dueling": conf["hyper_params"]["dueling"]}),

                      "eps": conf["eps"],

                      "action_size": self.action_size,
                      "state_size": self.state_size,

                      "memory_type": conf["memory"]["type"],
                      "memory_size": conf["memory"]["size"],
                      "beta_fn": beta_fn,

                      "optimizer": tf.train.RMSPropOptimizer(learning_rate=conf["hyper_params"]["lr"], momentum=0.95,
                                                             epsilon=0.01),
                      "discount": conf["hyper_params"]["discount"],

                      "net_update_size": conf["hyper_params"]["net_update_size"],
                      "net_update_frequency": conf["hyper_params"]["net_update_frequency"],

                      "double_Q": conf["hyper_params"]["double_Q"],

                      "delta_clip": 1,
                      "grad_norm_clip": 10,

                      "problem_type": conf["problem_type"],
                      "reload": conf["reload"],

                      "logdir": self.logdir,
                      "sess": conf["sess"],
                      }

        return train_dict

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
                total_beta_time: Positive integer - Number of steps to ascend beta towards 1,
                                 used only in case of memory type RankBased
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

    def _fill_memory(self):
        """
        Fills the memory before learning. This allows us to have higher diversity of training transitions thus
        lowers the bias of learning at the beginning.
        """
        while not self.memory.is_full():
            if self.memory_type != "RankBased" and len(self.memory) > 40000:
                break
            done = False
            s = self.env.reset()
            while not done:

                a = np.random.randint(self.action_size)
                s_, r, done, _ = self.env.step(a)

                done = float(done)

                transition = (s, a, r, s_, done)
                if self.memory_type == "RankBased":
                    self.memory.add_transition(transition, np.random.rand())
                else:
                    self.memory.add_transition(transition)

                s = s_

    def _load_memory(self, conf):
        """
        This function loads memory from saved directory

        returns: Memory max size and getter for beta and weight in case of RankBased memory type otherwise None
        """
        if self.memory_type == "RankBased":
            self.memory = MemoryRankBased(True, **conf)
            self.median_probability = self.memory.probabilities[int(self.memory.max_size // 2)]
            return self.memory.max_size, lambda: self.memory.get_beta_plus_weight(self.sess)
        else:
            self.memory = MemoryUniform(**conf["memory"])
            self._fill_memory()
            return conf["memory"]["size"], None

    def _train(self, transition):
        """
        Function that uses batch of examples with appended new transition to perform one training step.
        New transition is then added to the memory.
        args:
        transition: np.array - last transition (s,a,r,s_,d)
        """
        if self.memory_type == "RankBased":
            samples, probabilities = self.memory.get_n_samples()
            probabilities = np.append(probabilities, self.median_probability)
        else:
            samples = self.memory.get_n_samples()

        s, a, r, s_, d = samples

        t_s, t_a, t_r, t_s_, t_d = transition
        t_s, t_s_ = np.array(t_s, copy=False), np.array(t_s_, copy=False)

        s = np.append(s, t_s[None, ...], axis=0)
        s_ = np.append(s_, t_s_[None, ...], axis=0)
        a = np.append(a, [t_a], axis=0)
        r = np.append(r, [t_r], axis=0)
        d = np.append(d, [t_d], axis=0)

        s = np.stack(s, 0)
        s_ = np.stack(s_, 0)

        if self.memory_type == "RankBased":
            td = self.train_model.train(s, a, r, s_, d, probs=probabilities)
            self.memory.update(td[:-1])
            self.memory.add_transition(transition=transition, priority=td[-1])
        else:
            self.train_model.train(s, a, r, s_, d)

    @my_logger
    def solve(self,
              flag_container={"done": False, "show": False, "verbosity": 2},
              tune=False,
              total_length=20e6,
              target=999999,
              train_frequency=4,
              ):
        """
        Function consisting of the whole training loop: action performing -> state processing into transitions
        -> calling train function

        args:
        flag_container: dict - Can work as pointers to values changing behavior of function.
                        These values can be controlled via other thread.
            done: Bool - Whether we should save current state of learning and end
            show: Bool - Whether to render and show environment (slows learning)
            tune: Bool - Whether we tune hyper params with bayes optimization
            verbosity: Positive integer - How much verbose should function be

        returns: Mean reward of past 1000 episodes
        """

        if flag_container["verbosity"] > 0:
            # self._constant_time(np.array(self.env.reset()))
            print("\nLEARNING STARTED\n", )
        time.sleep(10)
        ep_counter = 0
        step_counter = self.train_model.get_glob_step() * train_frequency

        while True:
            time_start = time.time()

            steps = 0
            R = 0
            s = self.env.reset()

            done = False

            start_Qs = self.train_model.predict_one(np.array(s, copy=False))

            while not done:
                if flag_container["show"]:
                    self.env.render()

                    time.sleep(0.01)
                if step_counter % self.memory_size == 0 and self.memory_type == "RankBased":
                    if flag_container["verbosity"] > 0:
                        print("Memory sorting")
                    self.memory.heap_sort()
                a = self.train_model.act_epsilon(np.array(s, copy=False))
                s_, r, done, _ = self.env.step(a)

                transition = (s, a, r, s_, float(done))
                self.memory.add_transition(transition)
                if step_counter % train_frequency == 0:
                    self._train(transition)
                    if (step_counter // train_frequency) % 100000 == 0:
                        print("test time: ")
                        play_rew = self.play(30, flag_container)
                        print(play_rew)
                        self.train_model.update_play_rew(play_rew)

                steps += 1
                step_counter += 1
                R += r
                s = s_

            ep_counter += 1
            time_end = time.time()

            self._handle_episode_timer(time_start,
                                       time_end,
                                       verbosity=flag_container["verbosity"],
                                       )

            self.train_model.update_mean_ep_reward(R)

            if tune and self.train_model.get_mean_reward_per_ep_100() >= target:
                return self.train_model.get_mean_reward_per_ep_100()

            if tune and step_counter > total_length:
                return self.train_model.get_mean_reward_per_ep_1000()

            if flag_container["done"]:
                if tune:
                    return 0
                else:
                    return self.train_model.get_mean_reward_per_ep_1000()

            if flag_container["verbosity"] > 0:
                last_Qs = self.train_model.predict_one(np.array(s, copy=False))
                print("##################")
                print("It took ", time_end - time_start, "seconds from last print timeeee ")
                print("Average time per time-step is", (time_end - time_start) / steps)
                print("last Qs: ", last_Qs)
                print("problem start Qs: ", start_Qs)
                print("reward for ", ep_counter, "-th episode is : ", R)
                print("##################")

    def play(self,
             episodes=1,
             flag_container={"done": False, "show": True, "verbosity": 2},):
        """
        Test agent on the environment for n episodes and return average reaward per episode
        args:
            episodes: number of the episodes
            flag_container: dict - Can work as pointers to values changing behavior of function.
                                   These values can be controlled via other thread.
                done: Bool - Whether we should save current state of learning and end
                show: Bool - Whether to render and show environment (slows learning)
                tune: Bool - Whether we tune hyper params with bayes optimization
                verbosity: Positive integer - How much verbose should function be

        """
        reward_total = 0
        self.env_play.reset()
        lives_max = self.env_play.unwrapped.ale.lives()
        actual_episodes = episodes * lives_max
        if actual_episodes == 0:
            actual_episodes = episodes
        for _ in range(actual_episodes):
            s = self.env_play.reset()
            done = False
            while not done:
                if flag_container["show"]:
                    self.env_play.render()
                    time.sleep(0.01)
                a = self.train_model.act(s)
                s_, r, done, _ = self.env_play.step(a)
                s_ = np.array(s_, copy=False)
                reward_total += r
                s = s_

            if flag_container["done"]:
                break

        return reward_total / episodes
