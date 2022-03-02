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
        if cfg['PDE']['type'] == 'Burger':
            raise TypeError('The Burgers equation requires a two-dimensional grid!')
        if cfg['space']['grid_size']['x'] < 3:
            raise ValueError("Grid size should be at least 2! Increase grid size.")
    else:
        if cfg['space']['grid_size']['x'] < 3:
            raise ValueError("Grid size in dimension 1should be at least 2! Increase grid size.")
        if cfg['space']['grid_size']['y'] < 3:
            raise ValueError("Grid size in dimension 1should be at least 2! Increase grid size.")


# ... Useful function decorator ........................................................................................

def adapt_input(func=None, *, dtype=torch.float, requires_grad: bool = False,
                output_dim: int = 1):
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
                res = func(x.detach().clone().numpy(), *args, **kwargs)
                return torch.reshape(torch.tensor([res], dtype=dtype, requires_grad=requires_grad), (1,))

            # If x is a list of points
            else:
                x = np.resize([x.detach().clone().numpy()], (len(x), len(x[0])))
                res = [torch.tensor(func(val, *args, *kwargs), dtype=dtype, requires_grad=requires_grad) for val in x]
                return torch.reshape(torch.stack(res), (len(x), output_dim))

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
