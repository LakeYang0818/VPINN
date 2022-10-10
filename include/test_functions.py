from functools import partial
from typing import Any, Sequence, Union

import numpy as np
import xarray as xr
from scipy.special import binom, chebyt, gamma, jacobi

"""Test functions used in the VPINN model. The test functions are only defined on one-dimensional axes.
   Higher dimensional test functions are products of one-dimensional test functions."""


# ======================================================================================================================
#  ╔╦╗╔═╗╔═╗╔╦╗  ╔═╗╦ ╦╔╗╔╔═╗╔╦╗╦╔═╗╔╗╔╔═╗
#   ║ ║╣ ╚═╗ ║   ╠╣ ║ ║║║║║   ║ ║║ ║║║║╚═╗
#   ╩ ╚═╝╚═╝ ╩   ╚  ╚═╝╝╚╝╚═╝ ╩ ╩╚═╝╝╚╝╚═╝
# ======================================================================================================================


def test_function_1d(x: Any, index: int, *, d: int = 0, type: str) -> float:
    """The d-th derivative of a one-dimensional test function, evaluated at a single point.

    :param x: the point at which to evaluate the test function.
    :param index: the index of the test function
    :param d: the order of the derivative
    :param type: the type of the test function
    :return: the test function evaluated at that point
    """

    def _chebyshev_poly(x: Any, n: int, *, d: int) -> float:

        """The d-th derivative of the nth Chebyshev polynomial of the first kind (including for d=0)"""

        if d > n:
            return 0
        elif d == 0:
            return chebyt(n)(x)
        else:
            res = 0
            for i in range(n - d + 1):
                if (n - d) % 2 != i % 2:
                    continue
                A = binom((n + d - i) / 2 - 1, (n - d - i) / 2)
                B = gamma((n + d + i) / 2)
                C = gamma((n - d + i) / 2 + 1)
                D = _chebyshev_poly(x, i, d=0)
                v = A * B / C * D
                if i == 0:
                    v *= 1.0 / 2
                res += v
            return 2**d * n * res

    def _jacobi_poly(x: Any, n: int, *, d: int, a: int, b: int) -> float:

        """The dth-derivative of the Jacobi polynomial of order n (including for d = 0)"""

        if d == 0:
            return jacobi(n, a, b)(x)
        elif d > n:
            return 0
        else:
            return (
                gamma(a + b + n + 1 + d)
                / (2**d * gamma(a + b + n + 1))
                * jacobi(n - d, a + d, b + d)(x)
            )

    def _sin(x: Any, n: int, *, d: int) -> float:

        """The d-th derivative of the sine test function of order n (including for d = 0)"""
        if d % 4 == 0:
            return np.sin(n * np.pi * x)
        elif d % 4 == 1:
            return np.cos(n * np.pi * x)
        elif d % 4 == 2:
            return -1 * np.sin(n * np.pi * x)
        elif d % 4 == 3:
            return -1 * np.cos(n * np.pi * x)

    if type.lower() == "chebyshev":
        return _chebyshev_poly(x, index + 1, d=d) - _chebyshev_poly(x, index - 1, d=d)

    elif type.lower() == "legendre":
        return _jacobi_poly(x, index + 1, a=0, b=0, d=d) - _jacobi_poly(
            x, index - 1, a=0, b=0, d=d
        )
    elif type.lower() == "sine":
        return _sin(x, index, d=d)


def test_function_vec(x: Any, index: Sequence, *, d: Sequence = None, type: str):
    """Test function evaluation at an n-dimensional grid point.

    :param x: the point at which to evaluate the test function. Test functions are products of 1d test functions.
    :param index: the multiindex of the test function
    :param d: the multiindex of the derivative
    :param type: the type of the test function
    :return: the test function evaluated at that point
    """

    d = split_derivative_orders(d)

    return np.array(
        [
            np.prod(
                [
                    test_function_1d(x[i], index[i], d=order[i], type=type)
                    for i in range(len(x))
                ]
            )
            for order in d
        ]
    )


def split_derivative_orders(d: Sequence) -> Sequence[Sequence]:
    """Splits a multiindex of derivatives into a sequence of individual derivative orders, in order to iteratively
    calculate partial derivatives of products of test functions.
    Example: [1, 0, 3] -> [[1, 0, 0], [0, 0, 3]]

    :param d: the multiindex of the derivatives
    :return: the sequenced derivatives
    """
    if (np.array(d) == 0).all():
        return [np.array(d, dtype=int)]
    else:
        res = []
        for idx in np.nonzero(d)[0]:
            res.append(np.zeros(len(d), dtype=int))
            res[-1][idx] = d[idx]

        return res


# ======================================================================================================================
#  ╔═╗╦  ╦╔═╗╦  ╦ ╦╔═╗╔╦╗╦╔═╗╔╗╔  ╔═╗╦ ╦╔╗╔╔═╗╔╦╗╦╔═╗╔╗╔╔═╗
#  ║╣ ╚╗╔╝╠═╣║  ║ ║╠═╣ ║ ║║ ║║║║  ╠╣ ║ ║║║║║   ║ ║║ ║║║║╚═╗
#  ╚═╝ ╚╝ ╩ ╩╩═╝╚═╝╩ ╩ ╩ ╩╚═╝╝╚╝  ╚  ╚═╝╝╚╝╚═╝ ╩ ╩╚═╝╝╚╝╚═╝
# ======================================================================================================================


def tf_grid_evaluation(
    grid: Union[xr.DataArray, xr.Dataset],
    test_function_indices: xr.DataArray,
    *,
    type: str,
    d: int = 0,
    core_dim: str = "idx"
):
    """Efficient evaluation of the test functions on a grid. For high dimensional grids, it is
       sufficient to evaluate the test functions on the grid axes and then combine the test
       function values together, thereby significantly reducing the computational cost. This is necessary as the
       test functions themselves cannot be efficiently vectorised. The derivative order can be 0.

    :param grid: the grid points on which to evaluate the
    :param test_function_indices: the multi-indices of the test functions to evaluate
    :param type: the type of test function to use
    :param d: (optional) the order of the derivative (can be zero)
    :param core_dim (optional) the dimension over which to vectorise the operation. Typically, this is the coordinate
        index.
    :return: an xarray.DataArray of the test function values on the grid, indexed by the test function index and the
        grid coordinates.
    """

    def _efficient_evaluation(
        grid: Union[xr.DataArray, xr.Dataset],
        index: Union[int, Sequence],
        *,
        dim: int = 1,
        d: int = 0,
        type: str,
        core_dim: str = None
    ):

        """Evaluation routine for a single test function multi-index on a grid.

        :param grid: the grid on where to evaluate the test functions
        :param index: the multi-index of the test function to use
        :param d: the derivative order
        :param type: which test function kind to use
        :return: an xr.Dataset of test function indices and their values on the grid domain
        """

        # One-dimensional case: apply the test function to each grid coordinate
        if dim == 1:

            return xr.apply_ufunc(
                partial(
                    test_function_1d,
                    index=index,
                    type=type,
                    d=d,
                ),
                grid,
                input_core_dims=[[core_dim]] if core_dim else None,
                dask="allowed",
                vectorize=True,
                keep_attrs=True,
            )

        # Higher dimensional case: apply the test function to each axis and combine into a grid
        else:

            d = split_derivative_orders(np.ones(dim) * d)
            d_res = []

            # Calculate the derivatives along each axis
            for j, derivative_axis in enumerate(d):

                res = []
                for k, ax in enumerate(grid.attrs["space_dimensions"]):
                    # Get the test function values on each axis
                    res.append(
                        _efficient_evaluation(
                            grid.coords[ax],
                            index[k],
                            dim=1,
                            d=derivative_axis[k],
                            type=type,
                            core_dim=None,
                        ).data
                    )

                # Create a meshgrid of the values
                if dim == 2:

                    x, y = np.meshgrid(res[0], res[1])
                    data = np.stack([x, y], axis=-1)

                elif dim == 3:

                    x, y, z = np.meshgrid(res[0], res[1], res[2])
                    data = np.stack([x, y, z], axis=-1)

                # Combine into a xr.DataArray by stacking along the axes of partial differentiation
                d_res.append(
                    xr.DataArray(coords=grid.coords, data=data, dims=grid.dims)
                    .prod(dim="idx", keep_attrs=True)
                    .assign_coords(idx=j)
                    .expand_dims({"idx": [j]}, axis=-1)
                )

            stacked_derivatives = xr.concat(d_res, dim="idx")
            stacked_derivatives.attrs = grid.attrs

            return stacked_derivatives

    tf_labels = list(test_function_indices.coords)
    tf_labels.remove("idx")

    res = []
    for j, idx in enumerate(
        test_function_indices.data.reshape((-1, grid.attrs["grid_dimension"]))
    ):
        # Evaluate the test function on all the grid points, expanding the dataset to have the
        # test function indices as additional coordinates
        ds = _efficient_evaluation(
            grid,
            idx,
            dim=grid.attrs["grid_dimension"],
            d=d,
            type=type,
            core_dim=core_dim,
        )

        ds = ds.expand_dims(
            dim={tf_labels[i]: [val] for i, val in enumerate(idx)}
        ).stack(dim_name__0=(tf_labels))

        res.append(ds)

    # Merge the datasets
    res = xr.concat(res, dim="dim_name__0").unstack()

    # Add attributes
    res.attrs["test_function_dims"] = tf_labels

    # Reorder dimensions to ensure calling .data returns correct shape
    for idx in tf_labels:
        res = res.transpose(idx, ...)

    return res


def tf_simple_evaluation(
    grid: Union[xr.DataArray, xr.Dataset],
    test_function_indices: xr.DataArray,
    *,
    type: str,
    d: int = 0,
    core_dim: str = "idx"
):
    """Naive evaluation of the test functions on a grid. The test functions are simply applied to
       each grid point. This can be used for instance to evaluate the test functions on the boundary of a grid.
       The derivative order can be 0.

    :param grid: the grid points on which to evaluate the
    :param test_function_indices: the multi-indices of the test functions to evaluate
    :param type: the type of test function to use
    :param d: (optional) the order of the derivative (can be zero)
    :param core_dim (optional) the dimension over which to vectorise the operation. Typically, this is the coordinate
        index.
    :return: an xarray.DataArray of the test function values on the grid, indexed by the test function index and the
    grid coordinates.
    """

    tf_labels = list(test_function_indices.coords)
    tf_labels.remove("idx")

    res = []
    for j, idx in enumerate(
        test_function_indices.data.reshape((-1, grid.attrs["grid_dimension"]))
    ):
        # tf_res = [test_function_vec(pt, index=idx, d=d * np.ones(grid.attrs["grid_dimension"]),
        #                             type=type) for pt in grid.data]
        # ds = xr.Data
        # print(tf_res)
        # Evaluate the test function on all the grid points, expanding the dataset to have the
        # test function indices as additional coordinates
        ds = xr.apply_ufunc(
            partial(
                test_function_vec,
                index=idx,
                type=type,
                d=d * np.ones(grid.attrs["grid_dimension"]),
            ),
            grid,
            input_core_dims=[[core_dim]],
            dask="parallelized",
            vectorize=True,
            keep_attrs=True,
            output_core_dims=[["coords"]],
        )

        ds = ds.expand_dims(
            dim={tf_labels[i]: [val] for i, val in enumerate(idx)}
        ).stack(dim_name__0=(tf_labels))

        res.append(ds)

    # Merge the datasets
    res = xr.concat(res, dim="dim_name__0").unstack()

    # Add attributes
    res.attrs["test_function_dims"] = tf_labels

    # Transpose the Dataset to ensure calling .data later returns correct shape
    for idx in tf_labels:
        res = res.transpose(idx, ...)

    return res
