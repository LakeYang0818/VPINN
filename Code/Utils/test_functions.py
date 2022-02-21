"""Test functions used for calculating the residual loss"""
import numpy as np
from scipy.special import gamma
from scipy.special import roots_jacobi
import tensorflow as tf
from typing import Any, Sequence, Union

# Local imports
from .data_types import DataGrid, Grid
from Utils.functions import adapt_input, jacobi_poly, test_function_1d, dtest_function_1d


@adapt_input
def test_function(point: Union[float, Sequence[float]], n) -> float:
    """Returns the test function evaluated at a grid point of arbitrary
    dimension. Higher dimensional test functions are simply products of
    1d test functions.

    Args:
        point :Union[Sequence, float]: the point at which to evaluate the test function
        n: the number of the test function
    Returns:
        a list of function values on each grid point
    """
    if isinstance(point, Sequence):
        f = 1
        for coord in point:
            f *= test_function_1d(coord, n)

        return f
    else:
        return test_function_1d(point, n)


@adapt_input
def dtest_function(point: Union[float, Sequence[float]], n, *, d: int) -> Union[float, Sequence[float]]:
    """Returns a vector of partial of the nth partial derivatives of the test functions. Since the
    test functions are products of 1d test functions, the partial derivatives simplify considerably.
    Args:
        point :Union[Sequence, float]: the coordinate values
        n :int: the number of the test function
        d :int: the order of the derivatives
    Returns:
        a list of partial derivatives along each coordinate direction at each grid point. If there is only one
            coordinate dimension, the derivative of the function at that point is returned.
    """
    if isinstance(point, Sequence):
        res = []
        for i in range(len(point)):
            df_i = dtest_function_1d(point[i], n, d=d)
            for j in range(len(point)):
                if j == i:
                    continue
                else:
                    df_i *= test_function_1d(point[j], n)
            res.append(df_i)

        return res
    else:
        return dtest_function_1d(point, n, d=d)


def dtest_function_list(point: Sequence[Sequence[Any]], n, *, d: list) -> Sequence[Sequence[Any]]:
    """Calculates derivative of all orders from a given list for the test function at a point.
     This speeds things up considerably when both are required, as one only has to iterate over
     the grid once.

    Args:
        point :Union[Sequence, float]: the grid of coordinates
        n :int: the number of the test function
        d :list: a list of the order of the derivatives. If the list only has one item, the derivative of that order is
                 returned
    Returns:
        a list of lists of partial derivatives along each coordinate direction at each grid point.
    """
    res: list = []
    for order in d:
        df = []
        if isinstance(point, Sequence):
            for i in range(len(point)):
                df_i = dtest_function_1d(point[i], n, d=order)
                for j in range(len(point)):
                    if j == i:
                        continue
                    else:
                        df_i *= test_function_1d(point[j], n)
                df.append(df_i)
            res.append(df)
        else:
            df.append(dtest_function_1d(point, n, d=order))
        res.append(df)

    return res


def GaussLobattoJacobiWeights(q: int, *, a: int, b: int, dtype: tf.DType = tf.dtypes.float64) -> DataGrid:
    """Returns one-dimensional quadrature weights"""
    x = [tf.cast(r, dtype).numpy() for r in roots_jacobi(q-2, a+1, b+1)[0]]
    if a == 0 and b == 0:
        w = 2/((q-1)*q*(jacobi_poly(x, q-1, a=0, b=0)**2))
        w_l = 2/((q-1)*q*(jacobi_poly(-1, q-1, a=0, b=0)**2))
        w_r = 2/((q-1)*q*(jacobi_poly(1, q-1, a=0, b=0)**2))
    else:
        w = 2**(a+b+1)*gamma(a+q)*gamma(b+q)/((q-1)*gamma(q)*gamma(a+b+q+1)*(jacobi_poly(x, q-1, a=a, b=b)**2))
        w_l = (b+1)*2**(a+b+1)*gamma(a+q)*gamma(b+q)/((q-1)*gamma(q)*gamma(a+b+q+1)*(jacobi_poly(-1, q-1, a=a, b=b)**2))
        w_r = (a+1)*2**(a+b+1)*gamma(a+q)*gamma(b+q)/((q-1)*gamma(q)*gamma(a+b+q+1)*(jacobi_poly(1, q-1, a=a, b=b)**2))
    w = np.append(tf.cast(w, dtype).numpy(), tf.cast(w_r, dtype).numpy())
    w = np.append(tf.cast(w_l, dtype).numpy(), tf.cast(w, dtype).numpy())
    x = [tf.cast(-1, dtype).numpy()] + x + [tf.cast(1, dtype).numpy()]

    return DataGrid(x=Grid(x=x), f=w)
