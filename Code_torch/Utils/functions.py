import numpy as np
from scipy.special import gamma
from scipy.special import jacobi
import torch
from typing import Any, Sequence, Union

# Local imports
from .Types.DataSet import DataSet
from .Types.Grid import Grid
from .utils import adapt_input

"""Functions used in the VPINN model"""


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
    elif len(x) == 2:
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
        # return (0.1 * np.sin(2 * np.pi * x[0]) + np.tanh(10 * x[0])) * np.sin(2 * np.pi * x[1])

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
        return -amp * (0.1 * (omega ** 2) * np.sin(omega * x)
                       + (2 * r1 ** 2) * np.tanh(r1 * x) / np.cosh(r1 * x) ** 2)
    # 2D example
    elif len(x) == 2:
        return -2 * np.pi ** 2 * u(x)
        # A, B = 0.1, 10
        # return (np.sin(2 * np.pi * x[1]) * (
        #         -4 * np.pi ** 2 * A * np.sin(2 * np.pi * x[0]) - 2 * B ** 2 * np.tanh(B * x[0]) * np.cosh(
        #     B * x[0]) ** (-2))
        #         - 4 * np.pi ** 2 * np.sin(2 * np.pi * x[1]) * (A * np.sin(2 * np.pi * x[0]) + np.tanh(B * x[0])))

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
        return gamma(a + b + n + 1 + d) / (2 ** d * gamma(a + b + n + 1)) * jacobi_poly(x, n - d, a=a + d, b=b + d)


@adapt_input
def test_function(point: Any, n: int) -> float:
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


def dtest_function(point: Union[float, Sequence[float]], n: int, *, d: int, as_tensor: bool = True,
                   requires_grad: bool = False, dtype=torch.float) -> Any:
    """Returns a vector of partial of the nth partial derivatives of the test functions. Since the
    test functions are products of 1d test functions, the partial derivatives simplify considerably.
    Args:
        :param point: the coordinate values
        :param n: the index of the test function
        :param d: the order of the derivative
        :param as_tensor: whether to return a torch.Tensor
        :param requires_grad: whether the return value requires differentiation
    Returns:
        a list of partial derivatives along each coordinate direction at each grid point. If there is only one
            coordinate dimension, the derivative of the function at that point is returned.

    """

    if d == 0:
        return test_function(point, n)

    def _dtest_function_1d(x: Any, n: int, *, d: int) -> Union[float, Sequence[float]]:
        """Returns the derivative of the test function on a 1d grid"""
        return djacobi_poly(x, n + 1, d=d, a=0, b=0) - djacobi_poly(x, n - 1, d=d, a=0, b=0)

    res = []

    # To do: implement non-Tensor return type
    if as_tensor:
        # scalar
        if point.size() <= torch.Size([1]):
            return torch.tensor([_dtest_function_1d(point, n, d=d)], requires_grad=requires_grad, dtype=dtype)

        # single point with multiple coords
        elif point.dim() == 1:
            for i in range(len(point)):
                df_i = 1.0
                for j in range(len(point)):
                    if j == i:
                        df_i *= _dtest_function_1d(point[j], n, d=d)
                    else:
                        df_i *= test_function([point[j]], n)
                res.append(df_i)
            return torch.tensor(res, requires_grad=requires_grad, dtype=dtype)

        # list of points
        else:
            for p in point:
                res.append(dtest_function(p, n, d=d))
            return torch.reshape(torch.stack(res), (len(point), point.size()[-1]))


# ... Function utilities ...............................................................................................


def evaluate_test_funcs(grid: Grid, test_func_dim: Union[int, Sequence[int]], *, d: int = 0, output_dim: int = 1):
    """ Evaluate the test functions on a grid. For higher dimensional grid functions, it is
    sufficient to evaluate the test functions on the grid axes and then combine the test
    function values together.

    :param grid: the grid on which to evaluate the test functions
    :param test_func_dim: the number of test functions in each direction
    :param d: the derivative of the test function to use
    :param output_dim: the output dimension of the function
    :return: the test function values and test function indices
    """
    n_test_funcs: int = test_func_dim if grid.dim == 1 else test_func_dim[0] * test_func_dim[1]

    if grid.dim == 1:

        # Get the test function indices
        idx = [[i] for i in range(n_test_funcs)]

        # Evaluate the test functions on the grid interior, as they vanish on the boundary
        test_func_vals: Sequence = DataSet(coords=idx,
                                           data=[dtest_function(grid.interior, i, d=d) for i in
                                                 range(1, test_func_dim + 1)],
                                           as_tensor=True, requires_grad=False).data

    else:

        # Get the test function indices in x and y direction
        test_func_vals_x_0: Sequence = DataSet(coords=[[i] for i in range(test_func_dim[0])],
                                               data=[dtest_function(grid.x[1:-1], i, d=0) for i in
                                                     range(1, test_func_dim[0] + 1)],
                                               as_tensor=True, requires_grad=False).data
        test_func_vals_x: Sequence = DataSet(coords=[[i] for i in range(test_func_dim[0])],
                                             data=[dtest_function(grid.x[1:-1], i, d=d) for i in
                                                   range(1, test_func_dim[0] + 1)],
                                             as_tensor=True, requires_grad=False).data

        test_func_vals_y_0: Sequence = DataSet(coords=[[i] for i in range(test_func_dim[1])],
                                               data=[dtest_function(grid.y[1:-1], i, d=0) for i in
                                                     range(1, test_func_dim[1] + 1)],
                                               as_tensor=True, requires_grad=False).data
        test_func_vals_y: Sequence = DataSet(coords=[[i] for i in range(test_func_dim[1])],
                                             data=[dtest_function(grid.y[1:-1], i, d=d) for i in
                                                   range(1, test_func_dim[1] + 1)],
                                             as_tensor=True, requires_grad=False).data

        # Combine the indices
        idx: Sequence = Grid(x=[i for i in range(test_func_dim[0])], y=[i for i in range(test_func_dim[1])],
                             dtype=int, as_tensor=False).data.tolist()

        # Generate the grid of test function values
        if d == 0:
            test_func_vals: torch.Tensor = torch.from_numpy(np.array(
                [Grid(x=test_func_vals_x[idx[i][0]].numpy(),
                      y=test_func_vals_y[idx[i][1]].numpy(),
                      dtype=np.float32, as_tensor=False).data
                 for i in range(n_test_funcs)]))
            test_func_vals = torch.reshape(torch.prod(test_func_vals, dim=2),
                                           (n_test_funcs, len(grid.interior), 1))
        else:

            test_func_vals = []
            for k in idx:
                for y in range(len(grid.y[1:-1])):
                    for x in range(len(grid.x[1:-1])):
                        v = [test_func_vals_x[k[0]][x] * test_func_vals_y_0[k[1]][y],
                             test_func_vals_x_0[k[0]][x] * test_func_vals_y[k[1]][y]]
                        test_func_vals.append(torch.reshape(torch.stack(v), (2, )))

            test_func_vals = torch.reshape(torch.stack(test_func_vals), (n_test_funcs, len(grid.interior), 2))

        test_func_vals.requires_grad = False

    return test_func_vals, idx


def integrate(function_vals: Any, test_func_vals: Any, grid_vol: float, as_tensor: bool = True,
              dtype=torch.float, requires_grad: bool = False):
    """
    Integrates a function against a test function over a domain, using simple quadrature.
    :param function_vals: the function values on the domain.
    :param test_func_vals: the function values on the domain.
    :param grid_vol: the volume of the grid
    :param as_tensor: whether to return the values as a torch.Tensor
    :param dtype: the data type to use.
    :param requires_grad: whether the return values requires differentiation.
    :return: the value of the integral
    """
    if not as_tensor:
        return grid_vol / len(test_func_vals) * np.sum(function_vals * test_func_vals)
    else:
        res = grid_vol / len(test_func_vals) * torch.sum(function_vals * test_func_vals)
        if isinstance(res, torch.Tensor):
            return torch.reshape(res, (1,))
        else:
            return torch.reshape(torch.tensor(res, dtype=dtype, requires_grad=requires_grad), (1,))
