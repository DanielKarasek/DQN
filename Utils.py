import pickle

from skopt import dump
from skopt.callbacks import CheckpointSaver


class CheckpointSaverForLambdas(CheckpointSaver):
    """
    Inherited from skopt.callbacks.CheckpoinSaver class
    with changed Result object so it can also dump
    progress of bayesian hyperparams search where
    objective function is lambda function.
    Args & Kwargs are same as for original class
    """

    def __call__(self, res):
        res.specs["args"]["func"] = None
        dump(res, self.checkpoint_path, **self.dump_options)


class DoneCallback:
    """
    Currently callback class used for bayessian optimization
    Stops learning between evaluations of objective function
    if required by user. Note: Checkpoints stay so learning
    can be resumed

    Args:
        dict flag_dict: Dictionary containing boolean, which represents
                        whether the learning should stop or continue
    Kwargs:
        key: string,integer.. - Used as key under which boolean representing
                                whether to stop or not is saved in the dictionary
                                If nothing is passed default key("done") is used

    """

    def __init__(self, flag_dict, key=None):
        self.flag_dict = flag_dict
        self.key = "done" if key is None else key

    def __call__(self, result):
        return True if self.flag_dict[self.key] else False


def my_logger(func):
    """
    Decorator function used to log Args and Kwargs of given function

    Args:
        func: function which Args and Kwargs should be logged
    """
    import logging
    logging.basicConfig(filename="ParamsLogger.log", level=logging.INFO)

    def wrapper(*args, **kwargs):
        logging.info("Ran with args: {}, and kwargs: {}".format(args, kwargs))
        return func(*args, **kwargs)

    return wrapper


def stop_watch(func):
    """
    Decorator function to measure time from taken by function and prints this number out

    Args:
        func: function which time should be measured
    """
    import time

    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print("function {}, took {f:.4} seconds to finish".format(func.__name__, end - start))
        return func(args, kwargs)

    return wrapper


def unpickle(file):
    """
        Unpickles values from pickled file

        Args:
            file: String - Path to file which should be unpickled
    """
    with open(file, "rb") as myFile:
        return pickle.load(myFile)


def pickle_obj(obj, file):
    """
    Pickles object to given file destination

    Args:
        obj: Object to be pickled
        file: String - Path to file in which we wish to pickle
    """
    with open(file, "wb") as myFile:
        pickle.dump(obj, myFile)
