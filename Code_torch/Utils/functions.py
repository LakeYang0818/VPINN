from scipy.special import gamma
from scipy.special import jacobi
import numpy as np
import torch
from typing import Any, Union, Sequence

# Local imports
from .utils import adapt_input, adapt_output

"""Functions used in the VPINNS model"""

# ... Exact solution ...................................................................................................

@adapt_input
def u(x: Any) -> float:

    # The 1D function definition
    def _func_1d(val, as_tensor: bool):
        r1, omega, amp = 5, 4 * np.pi, 1
        if as_tensor:
            if isinstance(val, torch.Tensor):
                return amp * (0.1 * torch.sin(omega * val) + torch.tanh(r1 * val))
            else:
                return torch.tensor(amp * (0.1 * torch.sin(omega * val) + torch.tanh(r1 * val)))
        else:
            return amp * (0.1 * np.sin(omega * val) + np.tanh(r1 * val))

    # The 2D function definition
    def _func_2d(val, as_tensor: bool):
        res = 1
        for p in val:
            res *= _func_1d(p, as_tensor)
        return res

    return adapt_output(x, _func_1d, _func_2d)

# ... External forcing .................................................................................................

@adapt_input
def f(x: Any) -> float:

    # The 1D function definition
    def _func_1d(val, as_tensor: bool):
        r1, omega, amp = 5, 4 * np.pi, 1
        if as_tensor:
            if isinstance(val, torch.Tensor):
                return amp * (0.1 * (omega**2) * torch.sin(omega * val)
                            + (2*r1**2)*torch.tanh(r1 * val.clone().detach())/torch.cosh(r1*val)**2)
            else:
                return torch.tensor(amp * (0.1 * (omega**2) * torch.sin(omega * val)
                                        + (2*r1**2)*torch.tanh(r1 * val)/torch.cosh(r1*val)**2))
        else:
            return amp * (0.1 * (omega ** 2) * np.sin(omega * val)
                            + (2 * r1 ** 2) * np.tanh(r1 * val) / np.cosh(r1 * val)**2)

    # The 2D function definition
    def _func_2d(val, as_tensor: bool):
        res = 1
        for p in val:
            res *= _func_1d(p, as_tensor)
        return res

    return adapt_output(x, _func_1d, _func_2d)

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
def test_function_1d(x: Any, n) -> Union[float, Sequence[float]]:
    """Returns the test function used for a 1d grid"""
    return jacobi_poly(x, n+1, a=0, b=0) - jacobi_poly(x, n-1, a=0, b=0)


@adapt_input
def dtest_function_1d(x: Any, n: int, *, d: int) -> Union[float, Sequence[float]]:
    """Returns the derivative of the test function on a 1d grid"""
    return djacobi_poly(x, n+1, d=d, a=0, b=0) - djacobi_poly(x, n-1, d=d, a=0, b=0)


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
    if isinstance(point, torch.Tensor) or isinstance(point, Sequence):
        res = 1
        for coord in point:
            res *= test_function_1d(coord, n)

        if isinstance(point, torch.Tensor):
            return torch.reshape(torch.tensor([res]), (1,))
        else:
            return res
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
