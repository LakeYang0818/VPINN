import numpy as np
import torch
from typing import Any, Sequence

# Local imports
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
    :param dtype: the return data type
    :param requires_grad: whether the grid requires differentiation
    :return: the function output
    """

    def evaluate(x: Any, *args, **kwargs) -> Any:

        # Evaluate on torch.Tensors
        if isinstance(x, torch.Tensor):
            if x.dim() > 2:
                raise ValueError(f'Cannot evaluate functions on {x.dim()}-dimensional tensor!')

            # If x is a single point:
            elif x.dim() == 1:
                res = func(x.detach().clone().numpy(), *args, **kwargs)
                return torch.reshape(torch.tensor([res], dtype=dtype, requires_grad=requires_grad), (1,))

            # If x is a list of points
            else:
                x = np.resize([x.detach().clone().numpy()], (len(x), len(x[0])))
                res = [torch.tensor(func(val, *args, **kwargs), dtype=dtype, requires_grad=requires_grad) for val in x]
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


def rescale_grid(grid: Grid, *, as_tensor: bool = True, requires_grad: bool = False) -> Grid:
    """ Rescales a grid to the [-1, 1] x [-1, 1] interval

    :param grid: the grid to rescale.
    :param as_tensor: whether to return a torch.Tensor grid
    :param requires_grad: whether the rescaled grid requires gradients
    :return: the rescaled grid
    """

    if grid.dim == 1:
        return Grid(x=2 * (grid.x - grid.x[0]) / (grid.x[-1] - grid.x[0]) - 1,
                    as_tensor=as_tensor, requires_grad=requires_grad)

    elif grid.dim == 2:

        return Grid(x=2 * (grid.x - grid.x[0]) / (grid.x[-1] - grid.x[0]) - 1,
                    y=2 * (grid.y - grid.y[0]) / (grid.y[-1] - grid.y[0]) - 1,
                    as_tensor=as_tensor, requires_grad=requires_grad)


def integrate(function_vals: Any, test_func_vals: Any, grid_vol: float, as_tensor: bool = True,
              dtype=torch.float, requires_grad: bool = False):
    """
    Integrates a function against a test function over a domain, using simple quadrature.
    :param function_vals: the function values on the domain.
    :param test_func_vals: the function values on the domain.
    :param grid_vol: the volume of the grid
    :param as_tensor: whether to return the values as a torch.Tensor
    :param dtype: the data type to use.
    :param requires_grad: whether the return values requires differentiation.
    :return: the value of the integral
    """
    if not as_tensor:
        return grid_vol / len(test_func_vals) * np.sum(function_vals * test_func_vals)
    else:
        res = grid_vol / len(test_func_vals) * torch.sum(function_vals * test_func_vals)
        if isinstance(res, torch.Tensor):
            return torch.reshape(res, (1,))
        else:
            return torch.reshape(torch.tensor(res, dtype=dtype, requires_grad=requires_grad), (1,))
