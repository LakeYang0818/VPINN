from functools import partial
import numpy as np
import numbers
import torch
from typing import Any, Sequence

# Local imports
from .grid import Grid

"""Utility functions used in the VPINNS model"""

# ......................................................................................................................

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