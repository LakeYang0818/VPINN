import numbers
from typing import Any, Sequence, Union

import numpy as np
import torch
from scipy.special import binom, chebyt, gamma, jacobi

# Local imports
from .DataSet import DataSet
from .Grid import Grid, rescale_grid
from .utils import adapt_input

"""Functions used in the VPINN model"""


def chebyshev_poly(x: Any, n: int) -> Any:
    """Returns the nth Chebyshev polynomial of the first kind"""
    return chebyt(n)(x)


def dchebyshev_poly(x: Any, n: int, d: int) -> Any:
    """Return the d-th derivative of the nth Chebyshev polynomial of the first kind"""
    if d > n:
        return 0
    elif d == 0:
        return chebyshev_poly(x, n)
    else:
        res = 0
        for i in range(n - d + 1):
            if (n - d) % 2 != i % 2:
                continue
            A = binom((n + d - i) / 2 - 1, (n - d - i) / 2)
            B = gamma((n + d + i) / 2)
            C = gamma((n - d + i) / 2 + 1)
            D = chebyshev_poly(x, i)
            v = A * B / C * D
            if i == 0:
                v *= 1.0 / 2
            res += v
        return 2**d * n * res


def jacobi_poly(x: Any, n: int, *, a: int, b: int) -> Union[float, Sequence[float]]:
    """Returns the Jacobi polynomial of order n"""
    return jacobi(n, a, b)(x)


def djacobi_poly(
    x: Any, n: int, *, d: int, a: int, b: int
) -> Union[float, Sequence[float]]:
    """Returns the dth-derivative of the Jacobi polynomial of order n"""
    if d == 0:
        return jacobi_poly(x, n, a=a, b=b)
    elif d > n:
        return 0
    else:
        return (
            gamma(a + b + n + 1 + d)
            / (2**d * gamma(a + b + n + 1))
            * jacobi_poly(x, n - d, a=a + d, b=b + d)
        )


# How to normalise these?
@adapt_input
def test_function(point: Sequence, n: int, *, which: str = "Legendre") -> float:
    """Returns the test function evaluated at a grid point of arbitrary
    dimension. Higher dimensional test functions are simply products of
    1d test functions.

    :param point: the point at where to evaluate the function
    :param n: the index of the test function
    :return: the test function value at the point
    :raises ValueError: if the test function index is less than 1.
    """
    if n < 1:
        raise ValueError(
            f"Invalid test function index {n}; expected value greater than 0!"
        )

    if which == "Chebyshev":
        if isinstance(point, numbers.Number):
            return chebyshev_poly(point, n + 1) - chebyshev_poly(point, n - 1)
        else:
            res = 1
            for coord in point:
                res *= chebyshev_poly(coord, n + 1) - chebyshev_poly(coord, n - 1)
            return res

    elif which == "Legendre":

        if isinstance(point, numbers.Number):
            return jacobi_poly(point, n + 1, a=0, b=0) - jacobi_poly(
                point, n - 1, a=0, b=0
            )
        else:
            res = 1
            for coord in point:
                res *= jacobi_poly(coord, n + 1, a=0, b=0) - jacobi_poly(
                    coord, n - 1, a=0, b=0
                )
            return res

    elif which == "Sine":

        if isinstance(point, numbers.Number):
            return np.sin(n * np.pi * point)
        else:
            res = 1
            for coord in point:
                res *= np.sin(n * np.pi * coord)
            return res

    else:
        raise ValueError(f"Unrecognized test function type {which}!")


@adapt_input
def dtest_function(point: Any, n: int, *, d: int, which: str = "Legendre"):
    """Returns a vector of the partial derivatives of the nth test function. Since the
    test functions are products of 1d test functions, the partial derivatives simplify considerably.

    :param point: the coordinate values
    :param n: the index of the test function
    :param d: the order of the derivative
    :param as_tensor: whether to return a torch.Tensor
    :param requires_grad: whether the return value requires differentiation
    :return :
        a list of partial derivatives along each coordinate direction at each grid point. If there is only one
            coordinate dimension, the derivative of the function at that point is returned.

    """
    if d == 0:
        return adapt_input(test_function)(point, n, which=which)

    def _dtest_function_1d(x: Any) -> Union[float, Sequence[float]]:
        """Returns the derivative of the test function on a 1d grid"""
        if which == "Chebyshev":
            return dchebyshev_poly(x, n + 1, d=d) - dchebyshev_poly(x, n - 1, d=d)
        elif which == "Legendre":
            return djacobi_poly(x, n + 1, d=d, a=0, b=0) - djacobi_poly(
                x, n - 1, d=d, a=0, b=0
            )
        elif which == "Sine":
            if d % 4 == 0:
                return np.sin(n * np.pi * x)
            elif d % 4 == 1:
                return np.cos(n * np.pi * x)
            elif d % 4 == 2:
                return -1 * np.sin(n * np.pi * x)
            elif d % 4 == 3:
                return -1 * np.cos(n * np.pi * x)
        else:
            raise ValueError(f"Unrecognized test function type {which}!")

    res = []
    for i in range(len(point)):
        df_i = 1.0
        for j in range(len(point)):
            if j == i:
                df_i *= _dtest_function_1d(point[j])
            else:
                df_i *= test_function([point[j]], n, which=which)
        res.append(df_i)

    return res


# ... Function utilities ...............................................................................................


def testfunc_grid_evaluation(
    grid: Grid,
    test_func_dim: Union[int, Sequence[int]],
    *,
    d: int = 0,
    which: str = "Legendre",
    where: str = "interior",
) -> DataSet:
    """Efficiently evaluates the test functions on a grid. For high dimensional grids, it is
    sufficient to evaluate the test functions on the grid axes and then combine the test
    function values together, thereby significantly reducing the computational cost.

    :param grid: the grid on where to evaluate the test functions
    :param test_func_dim: the number of test functions in each direction
    :param d: the derivative order
    :param which: which test function kind to use
    :param where: where on the grid to evaluate the test function
    :return: a DataSet of test function indices and their values on the grid domain
    :raises ValueError: if an unrecognized 'which' argument is returned
    """
    if where not in ["interior", "boundary", "all"]:
        raise ValueError(f"Unrecognized parameter {where}!")

    # Get the total number of test functions
    n_test_funcs: int = (
        test_func_dim if grid.dim == 1 else test_func_dim[0] * test_func_dim[1]
    )

    # Rescale the grid to the domain of the test functions
    grid = rescale_grid(
        grid, new_domain=[-1, 1] if grid.dim == 1 else [[-1, 1], [-1, 1]]
    )

    values = []

    if grid.dim == 1:

        # Get the relevant grid domain
        if where == "interior":
            domain = (
                grid.interior if not grid.is_tensor else grid.interior.detach().clone()
            )
        elif where == "boundary":
            domain = (
                grid.boundary if not grid.is_tensor else grid.boundary.detach().clone()
            )
        elif where == "all":
            domain = grid.data if not grid.is_tensor else grid.data.detach().clone()

        domain_size = len(domain)

        # Get the test function indices
        indices = np.arange(1, n_test_funcs + 1, 1, dtype=int)

        if not grid.is_tensor:
            values = [dtest_function(domain, n, d=d, which=which) for n in indices]
        else:
            values = torch.stack(
                [dtest_function(domain, n, d=d, which=which) for n in indices]
            )

    # On two-dimensional grids, the test functions do not need to be evaluated on all grid points, but rather just
    # on the axes. This is computationally more efficient.
    elif grid.dim == 2:

        # Get the test function indices
        indices = Grid(
            x=[i for i in range(1, test_func_dim[0] + 1)],
            y=[i for i in range(1, test_func_dim[1] + 1)],
            dtype=int,
            as_tensor=False,
        )

        # Evaluate on the grid interior or the entire grid
        if where in ["interior", "all"]:

            if where == "interior":
                domain_x, domain_y = grid.x[1:-1], grid.y[1:-1]
            elif where == "all":
                domain_x, domain_y = grid.x, grid.y

            domain_size = len(domain_x) * len(domain_y)

            # Get the test function values on the x-axis
            vals_x_0: DataSet = DataSet(
                coords=indices.x,
                data=[
                    dtest_function(domain_x, idx[0], d=0, which=which)
                    for idx in indices.x
                ],
                as_tensor=grid.is_tensor,
                dtype=grid.dtype,
                requires_grad=grid.requires_grad,
            )

            # Get the test function derivative values on the x-axis
            vals_x_d: DataSet = DataSet(
                coords=indices.x,
                data=[
                    dtest_function(domain_x, idx[0], d=d, which=which)
                    for idx in indices.x
                ],
                as_tensor=grid.is_tensor,
                dtype=grid.dtype,
                requires_grad=grid.requires_grad,
            )

            # Get the test function values on the y-axis
            vals_y_0: DataSet = DataSet(
                coords=indices.y,
                data=[
                    dtest_function(domain_y, idx[0], d=0, which=which)
                    for idx in indices.y
                ],
                as_tensor=grid.is_tensor,
                dtype=grid.dtype,
                requires_grad=grid.requires_grad,
            )

            # Get the test function derivative values on the y-axis
            vals_y_d: DataSet = DataSet(
                coords=indices.y,
                data=[
                    dtest_function(domain_y, idx[0], d=d, which=which)
                    for idx in indices.y
                ],
                as_tensor=grid.is_tensor,
                dtype=grid.dtype,
                requires_grad=grid.requires_grad,
            )

            # If d=0 (no derivative), simply return a grid of the test function values. Test function values are
            # products of the axis values.
            if d == 0:
                if not grid.is_tensor:
                    # TO DO: this doesn't work
                    values = [
                        np.prod(
                            Grid(
                                x=vals_x_d.data[indices.data[i][0] - 1],
                                y=vals_y_d.data[indices.data[i][1] - 1],
                            ).data,
                            axis=1,
                        )
                        for i in range(n_test_funcs)
                    ]
                else:
                    values = torch.stack(
                        [
                            torch.prod(
                                Grid(
                                    x=vals_x_d.data[indices.data[i][0] - 1],
                                    y=vals_y_d.data[indices.data[i][1] - 1],
                                ).data,
                                dim=1,
                                keepdim=True,
                            )
                            for i in range(n_test_funcs)
                        ]
                    )

            # If a derivative is required, combine the test function values and their derivatives
            else:
                for index in indices:
                    for y in range(len(domain_y)):
                        for x in range(len(domain_x)):
                            v = [
                                vals_x_d.data[index[0] - 1][x]
                                * vals_y_0.data[index[1] - 1][y],
                                vals_x_0.data[index[0] - 1][x]
                                * vals_y_d.data[index[1] - 1][y],
                            ]
                            values.append(v)

                if grid.is_tensor:
                    values = torch.tensor(values)

        # On the boundary, the test functions can again only be evaluated on the axes, and the results then concatenated
        elif where == "boundary":

            domain_size = len(grid.boundary)

            # Test functions vanish on the boundary
            if d == 0:
                if not grid.is_tensor:
                    values = [0.0 for _ in range(domain_size * indices.size)]
                else:
                    values = [
                        torch.tensor(
                            [0.0], dtype=grid.dtype, requires_grad=grid.requires_grad
                        )
                        for _ in range(domain_size * indices.size)
                    ]

            else:
                for k in indices:

                    # Calculate the corner values
                    x_0, x_1 = dtest_function(
                        grid.x[0], k[0], d=d, which=which
                    ), dtest_function(grid.x[-1], k[0], d=d, which=which)
                    y_0, y_1 = dtest_function(
                        grid.y[0], k[1], d=d, which=which
                    ), dtest_function(grid.y[-1], k[1], d=d, which=which)

                    # Calculate the test function values on the boundary of the domain
                    if not grid.is_tensor:
                        lower = np.stack(
                            (
                                np.zeros(len(grid.x)),
                                test_function(grid.x, k[0], which=which) * y_0,
                            ),
                            1,
                        )
                        right = np.stack(
                            (
                                test_function(grid.y[1:-1], k[1], which=which) * x_1,
                                np.zeros(len(grid.y[1:-1])),
                            ),
                            1,
                        )
                        upper = np.stack(
                            (
                                np.zeros(len(grid.x)),
                                test_function(grid.x[::-1], k[0], which=which) * y_1,
                            ),
                            1,
                        )
                        left = np.stack(
                            (
                                test_function(
                                    grid.y[1:-1][::-1], k[1], which=which
                                ).flatten()
                                * x_0,
                                np.zeros(len(grid.y[1:-1])),
                            ),
                            1,
                        )
                    else:
                        lower = torch.stack(
                            [
                                torch.zeros(len(grid.x)),
                                torch.flatten(test_function(grid.x, k[0], which=which))
                                * y_0,
                            ],
                            dim=1,
                        )
                        right = torch.stack(
                            [
                                torch.flatten(
                                    test_function(grid.y[1:-1], k[1], which=which)
                                )
                                * x_1,
                                torch.zeros(len(grid.y[1:-1])),
                            ],
                            dim=1,
                        )
                        # Note: torch currently does not support the [::-1] syntax that can be used with numpy arrays.
                        upper = torch.stack(
                            [
                                torch.zeros(len(grid.x)),
                                torch.flatten(
                                    test_function(grid.x.flip(0), k[0], which=which)
                                )
                                * y_1,
                            ],
                            dim=1,
                        )
                        left = torch.stack(
                            [
                                torch.flatten(
                                    test_function(
                                        grid.y[1:-1].flip(0), k[1], which=which
                                    )
                                )
                                * x_0,
                                torch.zeros(len(grid.y[1:-1])),
                            ],
                            dim=1,
                        )

                    # Stack the boundary terms in such a way as to be consistent with the grid.boundary (contour of
                    # domain, oriented counterclockwise)
                    if not grid.is_tensor:
                        v = np.concatenate((lower, right, upper, left), 0)
                    else:
                        v = torch.cat((lower, right, upper, left), 0)

                    # Append to values
                    values.append(v)

            if grid.is_tensor:
                values = torch.stack(values)

    # Reshape the function values to appropriate shape
    out_dim = 1 if (d == 0 or grid.dim == 1) else 2

    if not grid.is_tensor:
        values = np.resize(values, (n_test_funcs, domain_size, out_dim))
    else:
        values = torch.reshape(values, (n_test_funcs, domain_size, out_dim))

    vals: DataSet = DataSet(
        coords=indices.data,
        data=values,
        as_tensor=grid.is_tensor,
        requires_grad=grid.requires_grad,
        dtype=grid.dtype,
    )

    return vals
