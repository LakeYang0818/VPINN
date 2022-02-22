from scipy.special import gamma
from scipy.special import jacobi
import numpy as np
import tensorflow as tf
from typing import Any, Union, Sequence
from functools import wraps

# Local imports
from Utils.data_types import Grid

# ....................................................................

# Decorator that prepares functions for different kinds of input (Grid type,
# sequence of floats, etc.) This way functions are all guaranteed to return
# the same data type


def adapt_input(func=None):
    def evaluate(x: Union[float, Sequence[Sequence[float]], Grid], *args, **kwargs) -> Union[float, Sequence[float]]:
        if isinstance(x, Sequence):
            if isinstance(x[0], Sequence):
                return [func(p, *args, **kwargs) for p in x]
            else:
                return func(x, *args, **kwargs)
        elif isinstance(x, Grid):
            return [func(p, *args, **kwargs) for p in x.data]
        else:
            return func(x, *args, **kwargs)

    return evaluate


# ... External functions .......................................................
# Exact solution
@adapt_input
def u(x: Any) -> float:
    # 2d case
    if isinstance(x, Sequence):
        res = 1
        for coord in x:
            r1 = 5
            omega = 4 * np.pi
            amp = 1
            utemp = 0.1 * np.sin(omega * coord) + np.tanh(r1 * coord)
            res *= amp * utemp
        return res
    else:
        r1 = 5
        omega = 4 * np.pi
        amp = 1
        utemp = 0.1 * np.sin(omega * x) + np.tanh(r1 * x)
        return amp * utemp



# External forcing
@adapt_input
def f(x: Any) -> float:
    if isinstance(x, Sequence):
        res = 1
        for coord in x:
            omega = 4 * np.pi
            r1 = 5
            amp = 1
            gtemp = -0.1 * (omega ** 2) * np.sin(omega * coord) - (2 * r1 ** 2) * (np.tanh(r1 * coord)) / (
                        (np.cosh(r1 * coord)) ** 2)
            res *= -amp * gtemp
        return res
    else:
        omega = 4 * np.pi
        r1 = 5
        amp = 1
        gtemp = -0.1 * (omega ** 2) * np.sin(omega * x) - (2 * r1 ** 2) * (np.tanh(r1 * x)) / ((np.cosh(r1 * x)) ** 2)
        return -amp*gtemp

# ... Test functions .......................................................
# Recursive generation of the Jacobi polynomial of order n
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
