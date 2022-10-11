import numbers
from functools import partial
from typing import Any, Sequence

import numpy as np
import torch
import xarray as xr


def integrate(
    func_1: torch.Tensor,
    func_2: torch.Tensor = None,
    *,
    domain_density: torch.Tensor,
    requires_grad: bool = True,
):
    """
    Integrates a function against another function over a domain, using simple quadrature. If the second function
    is not given, the first is simply integrated over the domain
    :param func_1: the function values on the domain.
    :param func_2: the function values on the domain. If none are passed, the function returns the integral of
        the function over the domain
    :param domain_density: the density of the grid
    :return: the value of the integral
    """
    if func_2 is None:

        func_2 = torch.ones_like(func_1)

    elif len(np.shape(func_1)) != len(np.shape(func_2)):
        raise ValueError(
            f"Function values and test function values must have same dimension, but are of "
            f"dimensions {len(np.shape(func_1))} and {len(np.shape(func_2))}."
        )

    # scalar case
    if len(func_1.size()) == 1 or func_1.size()[-1] == 1:

        return domain_density * torch.einsum("i..., i...->...", func_1, func_2)

    # vector-valued case: dot product

    elif func_1.size()[-1] == 2:

        return torch.sum(
            domain_density * torch.einsum("ij..., ij...->i...", func_1, func_2),
            dim=-1,
            keepdim=True,
        )


def integrate_xr(
    f: xr.DataArray, test_function_values: xr.DataArray, *, weights: xr.DataArray = None
) -> xr.DataArray:
    """Integrate a function over the interior of the grid"""

    # TODO use weights
    return test_function_values.attrs["grid_density"] * (test_function_values * f).isel(
        {val: slice(1, -1) for val in test_function_values.attrs["space_dimensions"]}
    ).sum(test_function_values.attrs["space_dimensions"])
