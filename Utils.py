import pickle


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
