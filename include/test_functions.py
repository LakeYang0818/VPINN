from functools import partial
import numpy as np
from scipy.special import binom, gamma, jacobi, chebyt
from typing import Any, Sequence, Union
import xarray as xr

"""Test functions used in the VPINN model"""


def chebyshev_poly(x: Any, n: int, d: int) -> Any:

    """Return the d-th derivative of the nth Chebyshev polynomial of the first kind"""

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
            D = chebyshev_poly(x, i, d=0)
            v = A * B / C * D
            if i == 0:
                v *= 1.0 / 2
            res += v
        return 2 ** d * n * res


def jacobi_poly(x: Any, n: int, *, d: int, a: int, b: int) -> Union[float, Sequence[float]]:

    """Returns the dth-derivative of the Jacobi polynomial of order n (including for d = 0)"""

    if d == 0:
        return jacobi(n, a, b)(x)
    elif d > n:
        return 0
    else:
        return gamma(a + b + n + 1 + d) / (2 ** d * gamma(a + b + n + 1)) * jacobi(n - d, a + d, b + d)(x)


def test_function(x: Any, index: Sequence, *, d: Sequence = None, type: str):

    """ The d-th derivative of the test function, evaluated at a single point.

    :param x: the point at which to evaluate the test function. Test functions are products of 1d test functions.
    :param index: the multiindex of the test function
    :param d: the multiindex of the derivative
    :param type: the type of the test function
    :return: the test function evaluated at that point
    """

    def split_derivative_orders(d: Sequence) -> Sequence[Sequence]:

        """ Splits a multiindex of derivatives into a sequence of individual derivative orders, in order to iteratively
        calculate partial derivatives of products of test functions.
        Example: [1, 0, 3] -> [[1, 0, 0], [0, 0, 3]]

        :param d: the multiindex of the derivatives
        :return: the sequenced derivatives
        """
        if np.array_equal(d, 0):
            return [d]
        else:
            res = []
            for idx in np.nonzero(d):
                res.append(np.zeros(len(d)))
                res[-1][idx] = d[idx]
            return res

    res = 0

    d = split_derivative_orders(d)

    for order in d:
        dim_res = 1
        for i in range(len(x)):
            if type.lower() == "chebyshev":
                dim_res *= chebyshev_poly(x[i], index[i] + 1, d=order[i]) - chebyshev_poly(x[i], index[i] - 1)

            elif type.lower() == "legendre":
                dim_res *= jacobi_poly(x[i], index[i] + 1, a=0, b=0, d=order[i]) - jacobi_poly(x[i], index[i] - 1, a=0, b=0, d=order[i])

            elif type.lower() == "sine":
                if order[i] % 4 == 0:
                    dim_res *= np.sin(index[i] * np.pi * x[i])
                elif order[i] % 4 == 1:
                    dim_res *= np.cos(index[i] * np.pi * x[i])
                elif order[i] % 4 == 2:
                    dim_res *= -1 * np.sin(index[i] * np.pi * x[i])
                elif order[i] % 4 == 3:
                    dim_res *= -1 * np.cos(index[i] * np.pi * x[i])
        res += dim_res

    return res


def evaluate_test_functions_on_grid(grid: xr.DataArray,
                                    test_function_indices: xr.DataArray,
                                    *,
                                    type: str,
                                    d: int = 0) -> xr.DataArray:

    """ Evaluates the d-th derivative of the test functions on a grid (d can be 0).

    :param grid: the grid points on which to evaluate the
    :param test_function_indices: the multi-indices of the test functions to evaluate
    :param type: the type of test function to use
    :param d: (optional) the order of the derivative (can be zero)
    :return:
    """

    tf_labels = list(test_function_indices.coords)
    tf_labels.remove('idx')

    res = []
    for j, idx in enumerate(test_function_indices.data.reshape((-1, grid.attrs['grid_dimension']))):

        # Evaluate the test function on all the grid points, expanding the dataset to have the
        # test function indices as additional coordinates
        ds = xr.apply_ufunc(partial(test_function,
                                    index=idx,
                                    type=type,
                                    d=d * np.ones(grid.attrs['grid_dimension'])),
                            grid, input_core_dims=[['idx']], dask='allowed', vectorize=True, keep_attrs=True)

        ds = ds.expand_dims(dim={tf_labels[i]: [val] for i, val in enumerate(idx)}).stack(dim_name__0=(tf_labels))

        res.append(ds)

    # Merge the datasets
    res = xr.concat(res, dim='dim_name__0').unstack()

    # Add attributes
    res.attrs['test_function_dims'] = tf_labels

    return res