import argparse
import threading

from builtins import bool
from os import makedirs

import gym
import tensorflow as tf

from Agent import Agent
from Wrappers import make_atari, wrap_deepmind

parser = argparse.ArgumentParser()

parser.add_argument("--env_id",
                    help="What is the representation of state:\nInteger 0,1,2 meaning - "
                         "underlying state of env(aka angles, speeds, etc.), RAM and RAW pixels respectively",
                    default="BreakoutNoFrameskip-v4",
                    type=str,
                    )

learn = parser.add_argument_group("Learn options")

learn.add_argument("--dueling",
                   help="Boolean - Whether to use dueling architecture",
                   type=bool,
                   default=False,
                   )

learn.add_argument("-Q",
                   "--double_Q",
                   help="Enables double  a Q implementation of deep Q net",
                   default=False,
                   type=bool,
                   )

learn.add_argument("-s",
                   "--net_update_size",
                   help="Moves targets NN weights closer towards train NN weights by factor of this number",
                   default=1,
                   type=float
                   )

learn.add_argument("-f",
                   "--net_update_frequency",
                   help="How often update weight of target (frozen) NN in case of double (D)DQN",
                   default=1000,
                   type=int
                   )

learn.add_argument("--layers",
                   help="Size of each (dense-activation always relu in between layers) layer in form of size,size,size... ",
                   default="124,64",
                   type=str,
                   )

learn.add_argument("-b",
                   "--batch_size",
                   help="Size of batches fed to neural net",
                   default=32,
                   type=int
                   )


learn.add_argument("-l",
                   "--learning_rate",
                   help="learning rate used for Neural network to use",
                   default=5e-5,
                   type=float
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


learn.add_argument("--play",
                   help="Boolean - Whether to play the game from learned/randomly(randomly if logdir is not passed)",
                   type=bool,
                   default=False
                   )


learn.add_argument("-d",
                   "--discount",
                   help="discount used to lower rewards further in future",
                   default=0.99,
                   type=float
                   )



memory_group = parser.add_argument_group("Memory options")

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

reload_group = parser.add_argument_group("Reload options")

reload_group.add_argument("-r",
                          "--reload",
                          help="Reloads",
                          type=bool,
                          default=False,
                          )
reload_group.add_argument("--logdir",
                          help="Target directory of paused learning",
                          type=str,
                          default="./",
                          )

args = parser.parse_args()

print(args)


def create_memory_dict():
    """
    Creates dictionary for memory from arguments given to the program
    """
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
    """
    Creates dictionary for 'hyper-params' from arguments given to the program
    """
    layers = args.layers.split(",")
    layers = [int(layer) for layer in layers]

    hyper_params_dict = {}

    hyper_params_dict["lr"] = args.learning_rate
    hyper_params_dict["layers"] = layers
    hyper_params_dict["discount"] = args.discount
    hyper_params_dict["net_update_size"] = args.net_update_size
    hyper_params_dict["net_update_frequency"] = args.net_update_frequency
    hyper_params_dict["double_Q"] = args.double_Q
    hyper_params_dict["dueling"] = args.dueling
    return hyper_params_dict


def create_eps_params():
    """
    Creates dictionary for epsilon from arguments given to the program
    """
    eps_dict = {}
    eps_dict["max"] = float(args.epsilon_max)
    eps_dict["min"] = float(args.epsilon_min)
    eps_dict["decay_steps"] = args.epsilon_decay_steps
    return eps_dict


def create_dict_agent_params(summary_dir,
                             env,
                             env_play,
                             sess,
                             problem_type,
                             reload):
    """
    Combines all dictionaries,logdir,environment and tf.session into one dictionary
    """
    params_dict = {}
    params_dict["memory"] = create_memory_dict()
    params_dict["hyper_params"] = create_hyper_params()
    params_dict["eps"] = create_eps_params()
    params_dict["logdir"] = summary_dir
    params_dict["env"] = env
    params_dict["env_play"] = env_play
    params_dict["sess"] = sess
    params_dict["problem_type"] = problem_type
    params_dict["reload"] = reload
    return params_dict


def load_env(summary_dir):
    """
    reloads name of environment
    args:
        summary_dir: directory in which file with the name(id) of environment is saved

    returns: String - name(id) of the environment

    """
    f = open(summary_dir + "/problem", "r")
    print(type(f))
    env_id = f.readline()
    f.close()
    return env_id


def automatic_dir_name():
    """
    returns: String - part of dir name containing values of basic hyperparameters
    """
    return "lr_" + str(args.learning_rate) +\
           "update_freq_" + str(args.net_update_frequency) +\
           "update_size_" + str(args.net_update_size) +\
           "batch_" + str(args.batch_size)


def create_summary_dir(env_id, summary_dir_path="./", add_automatic=True):
    """
    Creates summary dir with for the whole learning, adds inside the file with problem name(id)
    env_id: name(id) of an environment
    summary_dir_path: String - path which we wish the summary directory to have - default is a ./
    add_automatic: Boolean - Whether to add generated string (adding basic hyperparams into name)
                             to extend summary_dir_path
    returns: String - final relative path of the summary dir
    """
    if add_automatic:
        if summary_dir_path != "./":
            summary_dir_path += "_"
        summary_dir_path += automatic_dir_name()
        
    try:
        makedirs(summary_dir_path)
    except OSError:
        print("The directory %s already exists".format(summary_dir_path))

    f = open(summary_dir_path + "/problem", "w")
    f.write(env_id)
    f.close()
    return summary_dir_path


def make_env(env_id, env_type):
    """
    Creates the environment for learning and testing
    args:
        env_id: name of id
        env_type: type of OPEN AI env (retro, atari, classic etc...)
    returns: training environment, testing environment
    """
    if env_type == "atari":
        env = make_atari(env_id)
        env = wrap_deepmind(env, frame_stack=True)
        env_play = make_atari(env_id)
        env_play = wrap_deepmind(env_play, clip_rewards=False, frame_stack=True)
    else:
        env = gym.make(env_id)
        env_play = env
    return env, env_play


def get_env_prob_type(env_id):
    """
    Creates the environment from environment name (id)
    args: env_id: name (id) of environment
    returns: training environment, testing environment, OPEN AI env type
    """
    env_dict = {}
    for env in gym.envs.registry.all():
        env_dict[env.id] = env._entry_point[9:].split(":")[0]
    env, env_play = make_env(env_id, env_dict[env_id])
    return env, env_play, env_dict[env_id]


def run_play():
    """
    Loads pretrained agent and plays on the environment on which the agent was trained
    """
    if not args.logdir:
        print("no logdir passed")
        return None

    env_id = load_env(args.logdir)

    env, env_play, problem_type = get_env_prob_type(env_id=env_id)

    summary_dir = args.logdir

    flag_container = {"done": False, "show": False, "verbosity": 2}

    with tf.Session() as sess:
        params_dict = create_dict_agent_params(summary_dir, env, env_play, sess, problem_type, args.reload)
        agent = Agent(params_dict)
        print("Solving the problem...")
        done_thread = threading.Thread(target=lambda: input_flags(flag_container))
        done_thread.daemon = True
        done_thread.start()

        agent.play(n=100000, flag_container=flag_container)


def run_solve():
    # TODO: logdir None reload True, repair
    """
    Solves the given problem with parameters given. Both are from outside of the program
    """
    if args.reload:
        env_id = load_env(args.logdir)
        summary_dir = args.logdir
    else:
        env_id = args.env_id
        print(args.logdir)
        summary_dir = create_summary_dir(env_id=env_id,
                                         summary_dir_path=args.logdir,
                                         add_automatic=True,
                                         )

    env, env_play, problem_type = get_env_prob_type(env_id=env_id)

    tf_config = tf.ConfigProto()
    tf_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    flag_container = {"done": False, "show": False, "verbosity": 2}

    # with tf.Session(config=tf_config) as sess:
    with tf.Session() as sess:
        params_dict = create_dict_agent_params(summary_dir, env, env_play, sess, problem_type, args.reload)
        agent = Agent(params_dict)
        print("Solving the problem...")
        done_thread = threading.Thread(target=lambda: input_flags(flag_container))
        done_thread.daemon = True
        done_thread.start()

        agent.solve(flag_container=flag_container)


# TODO : (env,)
def _optimize_my_dear_bayes(agent_solve_dict,
                            env_id,
                            summary_dir="./",
                            conf=None,
                            update_conf_fn=None,
                            **optim_params):
    """
    deprecated function for bayesian
    optimization for given environment id
    """
    optim_params = optim_params["optim_params"]
    if not (conf or update_conf_fn):
        print("conf or update_conf_fn not passed")
        return

    update_conf_fn(conf, **optim_params)

    # Create different folders for different settings of DDQN and mem_type
    summary_dir += "lr_" + repr(round(conf["hyper_params"]["lr"], 5)) +\
                   "_batch_" + repr(conf["memory"]["batch_size"]) +\
                   "_layers_" + repr(conf["hyper_params"]["layers"]) +\
                   "_up_freq_" + repr(conf["hyper_params"]["net_update_frequency"]) +\
                   "_up_size_" + repr(round(conf["hyper_params"]["net_update_size"], 4))

    summary_dir = summary_dir.replace(" ", "")
    summary_dir = summary_dir.replace(",", "__")
    summary_dir = summary_dir.replace("[", "_")
    summary_dir = summary_dir.replace("]", "_")

    if "reload" not in conf.keys():
        conf["reload"] = False

    if conf["reload"]:
        env_id = load_env(summary_dir)
    else:
        create_summary_dir(env_id, summary_dir,)

    env, problem_type = get_env_prob_type(env_id=env_id)

    conf["env"] = env
    conf["logdir"] = summary_dir
    conf["problem_type"] = problem_type

    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    with tf.Session() as sess:
        conf["sess"] = sess
        agent = Agent(conf)
        agent_value = -agent.solve(**agent_solve_dict)

    tf.reset_default_graph()
    return agent_value


def input_flags(flag_container):
    """
        function running on its own thread, containing flag container to influence main thread during run
        flag_container: dictionary -
                        done: Boolean - Whether to save and exit
                        show: Boolean - Whether to render environment
                        Verbosity: int - How much should the main thread be

    """
    while True:
        try:
            a = input()
            a.upper()
            if a == 'Q':
                flag_container["done"] = True
            elif a == 'S':
                flag_container["show"] = False if flag_container["show"] else True
            elif a == 'V':
                num = int(input())
                flag_container["verbosity"] = num
        # wild except that shouldn't be used usually :O
        except:
            print("se neposer")


# TODO: add play to input flags^^ :)
def main():
    if args.command == "tune":
        pass #bayesian_optimization()
    elif args.command == "train":
        if args.play:
            run_play()
        else:
            run_solve()


if __name__ == "__main__":
    main()

