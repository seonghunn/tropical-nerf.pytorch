import warnings
import functools

from .tropical import *
# import tropical.utils.chamfer_distance
import tropical.torch_ext

import torch
torch.ext = tropical.torch_ext


def deprecated(reason=None):
    """
    This decorator can be used to mark functions as deprecated.
    It will result in a warning being emitted when the function is used.

    Parameters:
    reason (str): A message explaining why the function is deprecated.
                  If None or empty, a default message is used.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            message = f"Function '{func.__name__}' is deprecated."
            if reason:
                message += f" Reason: {reason}"
            warnings.warn(
                message,
                category=DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapped
    return decorator
