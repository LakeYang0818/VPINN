import numpy as np
import torch
from typing import Any, Sequence
from .Types.Grid import Grid

"""Utility functions used in the VPINNS model"""


# ......................................................................................................................

def validate_cfg(cfg: dict):
    """Checks the configuration settings are valid in order to prevent cryptic errors"""

    dim: int = cfg['space']['dimension']
    if dim not in {1, 2}:
        raise ValueError(f'Argument {dim} not supported! Dimension must be either 1 or 2!')
    if dim == 1:
        if isinstance(cfg['space']['grid_size'], Sequence):
            raise TypeError(f'For dimension {dim} the grid size must be a scalar! '
                            'Adjust the config.')
        if cfg['PDE']['type'] == 'Burger':
            raise TypeError('The Burgers equation requires a two-dimensional grid!')
        if isinstance(cfg['space']['grid_size'], Sequence):
            raise TypeError('The grid size should be a scalar! Adjust the config.')
        if cfg['space']['grid_size'] == 1:
            raise ValueError("Grid size should be at least 2! Increase grid size.")
    else:
        if not isinstance(cfg['space']['grid_size'], Sequence):
            raise TypeError(f'For dimension {dim} the grid size must be a sequence of the form [n_y, n_x]! '
                            'Adjust the config.')
        if 1 in cfg['space']['grid_size']:
            raise ValueError("Grid size should be at least 2! Increase grid size.")


# ... Function decorators ..............................................................................................

def adapt_input(func=None, *, output_dim: int = 1):
    """Decorator that ensures functions are able to return for different kinds of input.

    :param func: the function to be decorated
    :param output_dim: the functions output dimension. Is 1 (scalar function) by default
    :return: the function output
    """

    def evaluate(x: Any, *args, **kwargs) -> Any:

        # Evaluate on torch.Tensors
        if isinstance(x, torch.Tensor):
            if x.dim() > 2:
                raise ValueError(f'Cannot evaluate functions on {x.dim()}-dimensional tensor!')

            # If x is a single point:
            elif x.size() <= torch.Size([1]):
                return func(x, *args, **kwargs)

            # If x is a list of points
            else:
                return torch.reshape(
                    torch.stack([func(x[i], *args, **kwargs) for i in range(len(x))]),
                    (len(x), output_dim))

        # Evaluate on sequences that are not torch.Tensors
        elif isinstance(x, Sequence):
            if len(np.shape(x)) > 2:
                raise ValueError(f'Cannot evaluate functions on {len(np.shape(x))}-dimensional sequence!')

            # If x is a single point:
            elif len(np.shape(x)) <= 1:
                return func(x, *args, **kwargs)

            # If x is a list of points
            else:
                return np.resize(np.array([func(val, *args, **kwargs) for val in x]), (len(x), output_dim))

        # Evaluate on grids
        elif isinstance(x, Grid):
            return func(x.data, *args, **kwargs)

        # Evaluate functions on floats
        else:
            return func(x, *args, **kwargs)

    return evaluate


def adapt_output(x, func_1d, func_2d):
    """Decorator that ensures functions return correct return types that can be processed by the neural net.

    :param x: the x value at which the function is to be evaluated
    :param func_1d: the function if the input is a 1d point
    :param func_2d: the function for higher dimensional inputs
    :return: the function value at x
    """

    if isinstance(x, torch.Tensor):

        # Evaluation on a point with single coordinate
        if x.size() <= torch.Size([1]):
            return torch.reshape(func_1d(x, True), (1,))

        # Evaluation on a point with multiple coordinates
        else:
            return torch.reshape(func_2d(x, True), (1,))

    elif isinstance(x, Sequence):

        # Evaluation on a point with single coordinate
        if len(np.shape(x)) <= 1:
            return np.array([func_1d(x, False)])
        else:
            return np.array([func_2d(x, False)])
