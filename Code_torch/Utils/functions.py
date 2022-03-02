from scipy.special import gamma
from scipy.special import jacobi
import numpy as np
import torch
from typing import Any, Union, Sequence

# Import function decorator
from .utils import adapt_input

"""Functions used in the VPINNS model"""

# ... Exact solution ...................................................................................................

@adapt_input
def u(x: Any) -> float:
    """ The exact solution of the PDE. Make sure the function is defined for the dimension of the grid you
    are trying to evaluate. Currently only 1D and 2D grids are supported.

    :param x: the point at which to evaluate the PDE.
    :return: the function value at the point

    Raises:
        ValueError: if a point is passed with dimensionality for which the function is not defined
    """

    # Define the 1D case
    if len(x) == 1:
        r1, omega, amp = 5, 4 * np.pi, 1
        return amp * (0.1 * np.sin(omega * x) + np.tanh(r1 * x))

    # Define the 2D case
    elif len(x) == 2 :
        return np.sin(np.pi*x[0])*np.sin(np.pi*x[1])

    else:
        raise ValueError(f"You have not configured the function 'u' to handle {len(x)}-dimensional inputs!")

# ... External forcing .................................................................................................

@adapt_input
def f(x: Any) -> float:
    """ The external forcing of the PDE.Make sure the function is defined for the dimension of the grid you
    are trying to evaluate. Currently only 1D and 2D grids are supported.

    :param x: the point at which to evaluate the PDE.
    :return: the function value at the point

    Raises:
        ValueError: if a point is passed with dimensionality for which the function is not defined
    """

    # 1D example
    if len(x) == 1:
        r1, omega, amp = 5, 4 * np.pi, 1
        return amp * (0.1 * (omega ** 2) * np.sin(omega * x)
                      + (2 * r1 ** 2) * np.tanh(r1 * x) / np.cosh(r1 * x) ** 2)
    # 2D example
    elif len(x) == 2:
        return -2*np.pi**2*np.sin(np.pi*x[0])*np.sin(np.pi*x[1])

    else:
        raise ValueError(f"You have not configured the function 'f' to handle {len(x)}-dimensional inputs!")


# ... Test functions ...................................................................................................

def jacobi_poly(x: Any, n: int, *, a: int, b: int) -> Union[float, Sequence[float]]:
    """Returns the Jacobi polynomial of order n"""
    return jacobi(n, a, b)(x)


def djacobi_poly(x: Any, n: int, *, d: int, a: int, b: int) -> Union[float, Sequence[float]]:
    """Returns the dth-derivative of the Jacobi polynomial of order n"""
    if d == 0:
        return jacobi_poly(x, n, a=a, b=b)
    elif d >= n:
        return 0
    else:
        return gamma(a+b+n+1+d)/(2**d*gamma(a+b+n+1))*jacobi_poly(x, n-d, a=a+d, b=b+d)


@adapt_input
def test_function(point: Sequence, n) -> float:
    """Returns the test function evaluated at a grid point of arbitrary
    dimension. Higher dimensional test functions are simply products of
    1d test functions.

    :param point: the point at which to evaluate the function
    :param n: the index of the test function
    :return: the test function value at the point
    """

    res = 1
    for coord in point:
        res *= jacobi_poly(coord, n + 1, a=0, b=0) - jacobi_poly(coord, n - 1, a=0, b=0)

    return res


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

    def _dtest_function_1d(x: Any, n: int, *, d: int) -> Union[float, Sequence[float]]:
        """Returns the derivative of the test function on a 1d grid"""
        return djacobi_poly(x, n + 1, d=d, a=0, b=0) - djacobi_poly(x, n - 1, d=d, a=0, b=0)

    res = []
    for i in range(len(point)):
        df_i = _dtest_function_1d(point[i], n, d=d)
        for j in range(len(point)):
            if j == i:
                continue
            else:
                df_i *= _dtest_function_1d(point[j], n, d=d)
        res.append(df_i)

    return res


# ... Function integration .............................................................................................

def integrate(function_vals: Any, test_func_vals: Any, as_tensor: bool = True,
              dtype=torch.float, requires_grad: bool = False):
    """
    Integrates a function against a test function over a domain, using simple quadrature.
    :param function_vals: the function values on the domain.
    :param test_func_vals: the function values on the domain.
    :param as_tensor: whether to return the values as a torch.Tensor
    :param dtype: the data type to use.
    :param requires_grad: whether the return values requires differentiation.
    :return: the value of the integral
    """
    if not as_tensor:
        return 1.0 / len(test_func_vals) * np.sum(function_vals * test_func_vals)
    else:
        res = 1.0 / len(test_func_vals) * torch.sum(function_vals * test_func_vals)
        if isinstance(res, torch.Tensor):
            return torch.reshape(res, (1,))
        else:
            return torch.reshape(torch.tensor(res, dtype=dtype, requires_grad=requires_grad), (1,))
