import argparse
from builtins import bool
from os import mkdir
import threading

import gym
import numpy as np
from skopt import gp_minimize, load, dump
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import tensorflow as tf

from Agent import Agent
from Utils import DoneCallback, CheckpointSaverForLambdas

parser = argparse.ArgumentParser()

parser.add_argument("-p",
                    "--problem",
                    help="Problem to solve, 0-Cart Pole,1-Lunar Lander(discrete), 3-Mountain Car(discrete)",
                    choices=[0, 1, 2, 3, 4],
                    default=0,
                    type=int,
                    )

subparsers = parser.add_subparsers(help="Different methods for learning processes", dest='command')

tune = subparsers.add_parser("tune", help="Uses bayessian optimization for hyperparams tuning")
learn = subparsers.add_parser("learn", help="Uses your hyperparams for learning")

tune.add_argument("-Q",
                  "--DDQN",
                  help="Boolean - whether to solve via double deep Q net(DDQN) and RankBased memory, or just Deep Q net(DQN)",
                  type=bool,
                  default=True,
                  )
tune.add_argument("-r",
                  "--reload",
                  help="Reloads",
                  type=bool,
                  default=False,
                  )

tune.add_argument("--savedir",
                  help="Target directory of paused learning, required for TUNING",
                  type=str,
                  default=".",
                  )

tune.add_argument("-n",
                  "--n_calls",
                  help="How many calls different points in search find evaluate",
                  default="40",
                  type=int
                  )

tune.add_argument("-t",
                  "--memory_type",
                  help="Uniform or RankBased type of memory",
                  default="Uniform",
                  type=str
                  )

double_Q_group = learn.add_argument_group('Double Q options')

double_Q_group.add_argument("-Q",
                            "--double_Q",
                            help="Enables double  a Q implementation of deep Q net",
                            default=False,
                            type=bool,
                            )

double_Q_group.add_argument("-s",
                            "--net_update_size",
                            help="Moves targets NN weights closer towards train NN weights by factor of this number",
                            default=2e-4,
                            type=float
                            )

double_Q_group.add_argument("-f",
                            "--net_update_frequency",
                            help="How often update weight of target NN",
                            default=1,
                            type=int
                            )

learn.add_argument("-l",
                   "--learning_rate",
                   help="learning rate used for Neural network to use",
                   default=5e-5,
                   type=float
                   )

learn.add_argument("-d",
                   "--discount",
                   help="discount used to lower rewards further in future",
                   default=0.99,
                   type=float
                   )

learn.add_argument("--layers",
                   help="Size of each layer in form of size,size,size... ",
                   default="124,64",
                   type=str,
                   )

learn.add_argument("-b",
                   "--batch_size",
                   help="Size of batches fed to neural net",
                   default=32,
                   type=int
                   )

learn.add_argument("-e",
                   "--epsilon_min",
                   help="epsilon for epsilon-greedy policy",
                   default=0.05,
                   type=float
                   )

learn.add_argument("-E",
                   "--epsilon_max",
                   help="epsilon for epsilon-greedy policy",
                   default=1.,
                   type=float
                   )

learn.add_argument("-y",
                   "--epsilon_decay_steps",
                   help="Time steps needed to decay eps Max to eps Min",
                   default=50000,
                   type=int
                   )

memory_group = learn.add_argument_group("Memory options")

memory_group.add_argument("-m",
                          "--memory_size",
                          help="How many transitions we can store at once",
                          default=int(1e5),
                          type=int
                          )

memory_group.add_argument("-t",
                          "--memory_type",
                          help="Uniform or RankBased type of memory",
                          default="Uniform",
                          type=str
                          )

memory_group.add_argument("-B",
                          "--beta",
                          help="Beta of rank based memory",
                          default=0.6,
                          type=float,
                          )

memory_group.add_argument("-a",
                          "--alpha",
                          help="Alpha of rank based memory",
                          default=0.7,
                          type=float,
                          )

memory_group.add_argument("-L",
                          "--len_beta",
                          help="How many steps use to move beta to 1 in rank based memory",
                          default=3e5,
                          type=float,
                          )

reload_group = learn.add_argument_group("Reload options")

reload_group.add_argument("-r",
                          "--reload",
                          help="Reloads",
                          type=bool,
                          default=False,
                          )
reload_group.add_argument("--logdir",
                          help="Target directory of paused learning",
                          type=str,
                          default=None,
                          )

args = parser.parse_args()

print(args)


def create_memory_dict():
    '''
    Creates dictionary for memory from arguments given to the program
    '''
    memory_dict = {}
    memory_dict["size"] = args.memory_size
    memory_dict["batch_size"] = args.batch_size
    memory_dict["type"] = args.memory_type
    if memory_dict["type"] == "RankBased":
        memory_dict["alpha"] = args.alpha
        memory_dict["beta"] = args.beta
        memory_dict["total_beta_time"] = args.len_beta
    return memory_dict


def create_hyper_params():
    '''
    Creates dictionary for 'hyper-params' from arguments given to the program
    '''
    layers = args.layers.split(",")
    layers = [int(layer) for layer in layers]

    hyper_params_dict = {}

    hyper_params_dict["lr"] = args.learning_rate
    hyper_params_dict["layers"] = layers
    hyper_params_dict["discount"] = args.discount
    hyper_params_dict["net_update_size"] = args.net_update_size
    hyper_params_dict["net_update_frequency"] = args.net_update_frequency
    hyper_params_dict["double_Q"] = args.double_Q
    return hyper_params_dict


def create_eps_params():
    '''
    Creates dictionary for epsilon from arguments given to the program
    '''
    eps_dict = {}
    eps_dict["max"] = args.epsilon_max
    eps_dict["min"] = args.epsilon_min
    eps_dict["decay_steps"] = args.epsilon_decay_steps
    return eps_dict


def create_dict_agent_params(summary_dir, env, sess, ram, reload):
    '''
    Combines all dictionaries,logdir,environment and tf.session into one dictionary
    '''
    params_dict = {}
    params_dict["memory"] = create_memory_dict()
    params_dict["hyper_params"] = create_hyper_params()
    params_dict["eps"] = create_eps_params()
    params_dict["logdir"] = summary_dir
    params_dict["env"] = env
    params_dict["sess"] = sess
    params_dict["RAM"] = ram
    params_dict["reload"] = reload
    return params_dict


def run_solve():
    PROBLEMS = ["CartPole-v1", "LunarLander-v2", "MountainCar-v0", "SpaceInvaders-ram-v0"]

    if args.reload:
        summary_dir = args.logdir
        f = open(summary_dir + "/problem", "r")
        problem_num = int(f.read())
        f.close

    else:

        summary_dir = "alpha_" + str(args.alpha) + ",beta_" + str(args.beta) + ",lr_" + str(args.learning_rate)
        try:
            mkdir(summary_dir)
        except:
            pass

        problem_num = args.problem

        f = open(summary_dir + "/problem", "w")
        f.write(str(problem_num))
        f.close()

    ram = True if problem_num > 2 else False

    PROBLEM = PROBLEMS[problem_num]

    env = gym.make(PROBLEM)

    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    flag_container = {"done": False, "show": False, "tune": False}

    with tf.Session(config=config) as sess:
        params_dict = create_dict_agent_params(summary_dir, env, sess, ram, args.reload)
        agent = Agent(params_dict)
        print("Solving the problem...")
        done_thread = threading.Thread(target=lambda: input_flags(flag_container))
        done_thread.daemon = True
        done_thread.start()

        agent.solve(flag_container=flag_container)


def process_conf(memory_type, DDQN, **conf):
    hyper_params_dict = {}
    memory_dict = {}
    eps_dict = {}
    params_dict = {}

    hyper_params_dict["discount"] = 0.99

    eps_dict["max"] = 1
    eps_dict["min"] = 0.05
    params_dict["reload"] = False

    memory_dict["size"] = int(1e6)

    hyper_params_dict["lr"] = conf["lr"]
    hyper_params_dict["layers"] = conf["layers"]
    eps_dict["decay_steps"] = conf["decay_steps"]
    memory_dict["batch_size"] = conf["batch_size"]

    memory_dict["type"] = "Uniform"
    hyper_params_dict["double_Q"] = False
    hyper_params_dict["net_update_size"] = 0
    hyper_params_dict["net_update_frequency"] = 0
    memory_dict["alpha"] = 0
    memory_dict["beta"] = 0
    memory_dict["total_beta_time"] = 0

    if DDQN:
        hyper_params_dict["double_Q"] = True
        hyper_params_dict["net_update_size"] = conf["net_update_size"]
        hyper_params_dict["net_update_frequency"] = conf["net_update_frequency"]

    if memory_type == "RankBased":
        memory_dict["type"] = memory_type
        memory_dict["alpha"] = conf["alpha"]
        memory_dict["beta"] = conf["beta"]
        memory_dict["total_beta_time"] = conf["total_beta_time"]

    params_dict["eps"] = eps_dict
    params_dict["hyper_params"] = hyper_params_dict
    params_dict["memory"] = memory_dict

    return params_dict


def _optimize_my_dear_bayes(problem_num=3,
                            DDQN=True, memory_type="RankBased",
                            summary_dir="./",
                            flag_container={"done": False, "show": False, "tune": True},
                            **conf):
    conf = conf["conf"]
    # Make this more generalized in future

    layers = [conf["layer_0"], conf["layer_1"], conf["layer_2"]]
    conf["layers"] = []

    conf.pop("layer_0")
    conf.pop("layer_1")
    conf.pop("layer_2")

    for layer in layers:
        if layer == 0:
            break
        else:
            conf["layers"].append(layer)

    PROBLEMS = ["CartPole-v1", "LunarLander-v2", "MountainCar-v0", "SpaceInvaders-ram-v0", "Atlantis-ram-v0"]
    PROBLEM = PROBLEMS[problem_num]

    # Create different folders for different settings of DDQN and mem_type
    for k, v in zip(conf.keys(), conf.values()):
        if type(v) is np.float64:
            v = round(v, 6)
        summary_dir += str(k) + "_" + str(v) + "__"

    summary_dir = summary_dir.replace(" ", "")
    summary_dir = summary_dir.replace(",", "__")
    summary_dir = summary_dir.replace("[", "_")
    summary_dir = summary_dir.replace("]", "_")

    conf = process_conf(DDQN=DDQN,
                        memory_type=memory_type,
                        **conf,
                        )

    try:
        mkdir(summary_dir)
    except:
        pass

    f = open(summary_dir + "/problem", "w")
    f.write(str(problem_num))
    f.close()

    print("ahoj")
    env = gym.make(PROBLEM)
    print("ahoj")
    # debugger, find line which needs the enter :)

    conf["env"] = env
    conf["logdir"] = summary_dir
    conf["RAM"] = True if problem_num > 2 else False
    if flag_container["reload"]:
        conf["reload"] = True
        flag_container["reload"] = False

    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    with tf.Session(config=config) as sess:
        conf["sess"] = sess
        agent = Agent(conf)

        agent_value = -agent.solve(flag_container=flag_container)

    tf.reset_default_graph()
    return agent_value


def bayesian_optimization():
    np.set_printoptions(threshold=5)

    DDQN = args.DDQN
    problem_num = args.problem
    memory_type = args.memory_type
    n_calls = args.n_calls

    reload = args.reload
    savedir = args.savedir

    try:
        mkdir(savedir)
    except:
        pass

    checkdir = args.savedir + "/chekpoint.pkl"

    if reload:
        res = load(checkdir)
        x0 = res.x_iters
        y0 = res.func_vals[:-1]
        n_random_starts = 10 - len(x0)
        n_random_starts = 0 if n_random_starts < 0 else n_random_starts

    else:
        x0 = None
        y0 = None
        n_random_starts = 10

    lr_cs = Real(1e-6, 1e-2, name="lr")
    decay_steps_cs = Integer(0, 1000000, name="decay_steps")

    layer0_cs = Integer(5, 128, name="layer_0")
    layer1_cs = Integer(0, 64, name="layer_1")
    layer2_cs = Integer(0, 64, name="layer_2")

    update_size_cs = Real(0.0, 1.0, name="net_update_size")
    update_freq_cs = Integer(5, 10000, name="net_update_frequency")

    alpha_cs = Real(0.2, 1.0, name="alpha")
    beta_cs = Real(0.0, 1.0, name="beta")
    batch_size_cs = Integer(16, 256, name="batch_size")
    total_beta_cs = Integer(50000, 2000000, name="total_beta_time")

    flag_container = {"done": False, "show": False, "tune": True, "reload": reload}

    done_thread = threading.Thread(target=lambda: input_flags(flag_container))
    done_thread.daemon = True
    done_thread.start()

    is_done = DoneCallback(flag_container)
    saver = CheckpointSaverForLambdas(checkdir)

    callbacks = [saver, is_done]

    space = [lr_cs, layer0_cs, layer1_cs, layer2_cs, batch_size_cs, decay_steps_cs]

    if DDQN:
        space.append(update_size_cs)
        space.append(update_freq_cs)

    if memory_type == "RankBased":
        space.append(alpha_cs)
        space.append(beta_cs)
        space.append(total_beta_cs)

    # Add closure and remove need for custom saver
    function = lambda **x: _optimize_my_dear_bayes(DDQN=DDQN,
                                                   problem_num=problem_num,
                                                   memory_type=memory_type,
                                                   summary_dir=savedir + "/",
                                                   flag_container=flag_container,
                                                   conf=x,
                                                   )
    f_tmp = use_named_args(space)
    function = f_tmp(function)

    result = gp_minimize(func=function,
                         dimensions=space,
                         x0=x0,
                         y0=y0,
                         n_calls=n_calls,
                         n_random_starts=n_random_starts,
                         callback=callbacks,
                         )

    dump(res=result,
         filename=checkdir
         )


def input_flags(flag_container):
    while True:
        a = input()
        if a == 'Q':
            flag_container["done"] = True
        elif a == 'S':
            flag_container["show"] = False if flag_container["show"] else True


def main():
    if args.command == "tune":
        bayesian_optimization()
    elif args.command == "solve":
        run_solve()


if __name__ == "__main__":
    main()
