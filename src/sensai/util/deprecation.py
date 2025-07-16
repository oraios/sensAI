import warnings
import logging
from functools import wraps

log = logging.getLogger(__name__)


def deprecated(message):
    """
    This is a decorator for functions to mark them as deprecated, issuing a warning when the function is called

    :param message: message, which can indicate the new recommended approach or reason for deprecation
    :return: decorated function
    """
    def deprecated_decorator(func):
        @wraps(func)
        def deprecated_func(*args, **kwargs):
            func_name = func.__name__
            if func_name == "__init__":
                class_name = func.__qualname__.split('.')[0]
                msg = "{} is a deprecated class. {}".format(class_name, message)
            else:
                msg = "{} is a deprecated function. {}".format(func_name, message)
            if logging.Logger.root.hasHandlers():
                log.warning(msg)
            else:
                warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
                warnings.simplefilter('default', DeprecationWarning)
            return func(*args, **kwargs)
        return deprecated_func
    return deprecated_decorator
