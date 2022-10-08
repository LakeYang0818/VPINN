from functools import partial
import numpy as np
import numbers
import torch
from typing import Any, Sequence

import xarray as xr


def integrate(func_1: Any, func_2: Any = None, *, domain_density: float = 1, as_tensor: bool = True,
              requires_grad: bool = False):
    """
    Integrates a function against another function over a domain, using simple quadrature. If the second function
    is not given, the first is simply integrated over the domain
    :param func_1: the function values on the domain.
    :param func_2: the function values on the domain. If none are passed, the function returns the integral of
        the function over the domain
    :param domain_density: the density of the grid
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

            return domain_density * np.einsum('i..., i...->...', func_1, func_2)

        elif np.shape(func_1)[-1] == 2:
            return domain_density * np.einsum('ij..., ij...->...', func_1, func_2)

    else:

        # scalar case
        if len(func_1.size()) == 1 or func_1.size()[-1] == 1:

            res = domain_density * torch.einsum('i..., i...->...', func_1, func_2)

        # vector valued case
        elif func_1.size()[-1] == 2:

            res = domain_density * len(func_2) * torch.einsum('ij..., ij...->...', func_1, func_2)

        if isinstance(res, torch.Tensor):

            return torch.reshape(res, (-1, 1))
        else:

            return torch.reshape(torch.tensor(res, dtype=torch.float, requires_grad=requires_grad), (1,))


def integrate_xr(f: xr.DataArray, test_function_values: xr.DataArray, *, weights: xr.DataArray = None) -> xr.DataArray:
    """ Integrate a function over the interior of the grid """

    # TODO use weights
    return test_function_values.attrs['grid_density'] * \
        (f * test_function_values).isel(
           {val: slice(1, -1) for val in test_function_values.attrs['space_dimensions']}
        ).sum(test_function_values.attrs['space_dimensions'])
