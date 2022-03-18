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


def integrate(func_1: Any, func_2: Any = None, *, domain_volume: float = 1, as_tensor: bool = True,
              requires_grad: bool = False):
    """
    Integrates a function against another function over a domain, using simple quadrature. If the second function
    is not given, the first is simply integrated over the domain
    :param func_1: the function values on the domain.
    :param func_2: the function values on the domain. If none are passed, the function returns the integral of
        the function over the domain
    :param domain_vol: the volume of the domain
    :param as_tensor: whether to return the values as a torch.Tensor
    :param requires_grad: whether the return values requires differentiation.
    :return: the value of the integral
    """

    if func_2 is None:

        func_2 = torch.ones_like(func_1) if as_tensor else np.ones_like(func_1)

    elif len(np.shape(func_1)) != len(np.shape(func_2)):
        raise ValueError(f"Function values and test function values must have same dimension, but are of "
                         f"dimensions {len(np.shape(func_1))} and {len(np.shape(func_2))}.")

    if not as_tensor:
        if len(np.shape(func_1)) == 1 or np.shape(func_1)[-1] == 1:

            return domain_volume / len(func_2) * np.einsum('i..., i...->...', func_1, func_2)

        elif np.shape(func_1)[-1] == 2:
            return domain_volume / len(func_2) * np.einsum('ij..., ij...->...', func_1, func_2)

    else:

        # scalar case
        if len(func_1.size()) == 1 or func_1.size()[-1] == 1:

            res = domain_volume / len(func_2) * torch.einsum('i..., i...->...', func_1, func_2)

        # vector valued case
        elif func_1.size()[-1] == 2:

            res = domain_volume / len(func_2) * torch.einsum('ij..., ij...->...', func_1, func_2)

        if isinstance(res, torch.Tensor):

            return torch.reshape(res, (1,))

        else:

            return torch.reshape(torch.tensor(res, dtype=torch.float, requires_grad=requires_grad), (1,))



