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
                return torch.reshape(torch.tensor([res], dtype=dtype, requires_grad=requires_grad), (1, ))

            # If x is a single point:
            elif x.dim() == 1:
                res = func(x.detach().clone().numpy(), *args, **kwargs)
                return torch.reshape(torch.tensor([res], dtype=dtype, requires_grad=requires_grad),
                                     (int(np.prod(np.shape(res))), ))

            # If x is a list of points
            else:
                x = np.resize([x.detach().clone().numpy()], (len(x), len(x[0])))
                res = [torch.tensor(func(val, *args, **kwargs), dtype=dtype, requires_grad=requires_grad) for val in x]
                return torch.reshape(torch.stack(res), (int(len(res)), int(np.prod(np.shape(res[0])))))


        # Evaluate on sequences that are not torch.Tensors
        elif isinstance(x, Sequence) or isinstance(x, np.ndarray):
            if len(np.shape(x)) > 2:
                raise ValueError(f'Cannot evaluate functions on {len(np.shape(x))}-dimensional sequence!')

            # If x is a single point:
            elif len(np.shape(x)) <= 1:
                return func(x, *args, **kwargs)

            # If x is a list of points
            else:
                res = np.array([func(val, *args, **kwargs) for val in x])
                return np.resize(res, (int(len(res)), int(np.prod(np.shape(res[0])))))

        # Evaluate on grids
        elif isinstance(x, Grid):
            return adapt_input(func)(x.data, *args, **kwargs)

        # Evaluate functions on numbers
        elif isinstance(x, numbers.Number):
            return func(x, *args, **kwargs)

    return evaluate


def integrate(function_vals: Any, test_func_vals: Any = None, domain_vol: float = 1, as_tensor: bool = True,
              dtype=torch.float, requires_grad: bool = False):
    """
    Integrates a function against a test function over a domain, using simple quadrature.
    :param function_vals: the function values on the domain.
    :param test_func_vals: the function values on the domain. If none are passed, the function returns the integral of
        the function over the domain
    :param domain_vol: the volume of the domain
    :param as_tensor: whether to return the values as a torch.Tensor
    :param dtype: the data type to use.
    :param requires_grad: whether the return values requires differentiation.
    :return: the value of the integral
    """

    if not as_tensor:
        if np.shape(function_vals) != np.shape(function_vals):
            raise ValueError(f"Function values and test function values must have same shape, but are of "
                             f"shapes {np.shape(function_vals)} and {np.shape(function_vals)}.")
        if test_func_vals is None:
            test_func_vals = np.array([[1.0] for _ in range(len(function_vals))])

        return domain_vol / len(test_func_vals) * np.einsum('i..., i...->...', function_vals, test_func_vals)

    else:
        if function_vals.dim() != test_func_vals.dim():
            raise ValueError(f"Function values and test function values must have same dimension, but are of "
                             f"dimensions {function_vals.dim()} and {test_func_vals.dim()}.")
        if test_func_vals is None:
            test_func_vals = torch.reshape(torch.ones(len(function_vals)), (len(function_vals), 1))

        res = domain_vol / len(test_func_vals) * torch.einsum('i..., i...->...', function_vals, test_func_vals)

        if isinstance(res, torch.Tensor):

            return torch.reshape(res, (1,))

        else:

            return torch.reshape(torch.tensor(res, dtype=dtype, requires_grad=requires_grad), (1,))
