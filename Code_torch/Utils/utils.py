from functools import partial
import numpy as np
import numbers
import torch
from typing import Any, Sequence

# Local imports
from .Datatypes.Grid import Grid

"""Utility functions used in the VPINNS model"""


# ......................................................................................................................

def validate_cfg(cfg: dict):
    """Checks the configuration settings are valid in order to prevent cryptic errors"""

    dim: int = cfg['space']['dimension']
    if dim not in {1, 2}:
        raise ValueError(f'Argument {dim} not supported! Dimension must be either 1 or 2!')
    if dim == 1:
        if cfg['PDE']['type'] in ['Burger', 'PorousMedium']:
            raise TypeError(f"The {cfg['PDE']['type']} equation requires a two-dimensional grid!")
        if cfg['space']['grid_size']['x'] < 3:
            raise ValueError("Grid size should be at least 2! Increase grid size.")
    else:
        if cfg['space']['grid_size']['x'] < 3:
            raise ValueError("Grid size in dimension 1should be at least 2! Increase grid size.")
        if cfg['space']['grid_size']['y'] < 3:
            raise ValueError("Grid size in dimension 1should be at least 2! Increase grid size.")


# ... Useful function decorator ........................................................................................

def adapt_input(func=None, *, dtype=torch.float, requires_grad=False):
    """Decorator that ensures functions are able to return for different kinds of input.

    :param func: the function to be decorated
    :param output_dim: the function output dimension. Is 1 (scalar function) by default
    :param dtype: the return data type
    :param requires_grad: whether the grid requires differentiation
    :return: the function output
    """

    def evaluate(x: Any, *args, **kwargs) -> Any:

        # Evaluate on torch.Tensors
        if isinstance(x, torch.Tensor):

            if x.dim() > 2:
                raise ValueError(f'Cannot evaluate functions on {x.dim()}-dimensional tensor!')

            # If x is a scalar
            elif x.dim() == 0:
                res = func([x.detach().clone().numpy()], *args, **kwargs)
                return torch.tensor([res], dtype=dtype, requires_grad=requires_grad)

            # If x is a single point:
            elif x.dim() == 1:
                res = func(x.detach().clone().numpy(), *args, **kwargs)
                if isinstance(res, Sequence) or isinstance(res, np.ndarray):
                    return torch.reshape(torch.tensor([res], dtype=dtype, requires_grad=requires_grad),
                                        (1, len(res)))
                else:
                    return torch.reshape(torch.tensor([res], dtype=dtype, requires_grad=requires_grad), (len(res), ))

            # If x is a list of points
            else:
                x = np.resize([x.detach().clone().numpy()], (len(x), len(x[0])))
                res = [torch.tensor(func(val, *args, **kwargs), dtype=dtype, requires_grad=requires_grad) for val in x]
                if res[0].dim() == 0:
                    return torch.reshape(torch.stack(res), (len(res), 1))
                else:
                    return torch.reshape(torch.stack(res), (len(res), len(res[0])))

        # Evaluate on sequences that are not torch.Tensors
        elif isinstance(x, Sequence) or isinstance(x, np.ndarray):
            if len(np.shape(x)) > 2:
                raise ValueError(f'Cannot evaluate functions on {len(np.shape(x))}-dimensional sequence!')

            # If x is a single point:
            elif len(np.shape(x)) <= 1:
                return func(x, *args, **kwargs)

            # If x is a list of points
            else:
                return np.array([func(val, *args, **kwargs) for val in x])

        # Evaluate on grids
        elif isinstance(x, Grid):
            if isinstance(x.data, torch.Tensor):
                res = torch.tensor([func(val, *args, **kwargs) for val in x.data.detach().clone()], dtype=dtype,
                                   requires_grad=requires_grad)
                if res.dim()>1:
                    return torch.reshape(res, (len(x.data), res.size()[-1]))
                else:
                    return torch.reshape(res, (len(x.data), ))
            else:
                return np.resize(func([x.data], *args, **kwargs), (len(x.data), ))

        # Evaluate functions on numbers
        elif isinstance(x, numbers.Number):
            return func([x], *args, **kwargs) # This should not require an array structure

    return evaluate


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
        if np.shape(function_vals) != np.shape(function_vals):
            raise ValueError(f"Function values and test function values must have same shape, but are of "
                             f"shapes {np.shape(function_vals)} and {np.shape(function_vals)}.")
        return grid_vol / len(test_func_vals) * np.sum(function_vals * test_func_vals)
    else:
        if function_vals.dim() != test_func_vals.dim():
            raise ValueError(f"Function values and test function values must have same dimension, but are of "
                             f"dimensions {function_vals.dim()} and {test_func_vals.dim()}.")
        res = grid_vol / len(test_func_vals) * torch.sum(function_vals * test_func_vals)
        if isinstance(res, torch.Tensor):
            return torch.reshape(res, (1,))
        else:
            return torch.reshape(torch.tensor(res, dtype=dtype, requires_grad=requires_grad), (1,))
