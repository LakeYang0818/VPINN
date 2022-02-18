from typing import Sequence, Union
import tensorflow as tf
import numpy as np
from pyDOE import lhs

from .data_types import DataGrid, Grid
from .test_functions import GaussLobattoJacobiWeights


def get_grid(boundary: Sequence[float], grid_size: int, *, dtype: tf.DType = tf.dtypes.float64) -> Grid:
    """Constructs a one-dimensional grid.
    Args:
        boundary :Sequence: the boundaries of the grid
        grid_size :int: the number of grid points
        dtype: the datatype of the grid points
    Returns:
        a grid of the given values
    """
    lower = boundary[0]
    upper = boundary[1]
    step_size = (1.0*upper - lower) / (1.0*grid_size - 1)
    x = [tf.cast(lower + _ * step_size, dtype).numpy() for _ in range(grid_size)]
    return Grid(x=x)


def construct_grid(dim: int,
                   boundary: Sequence,
                   grid_size: Union[int, Sequence[int]],
                   *, dtype: tf.DType = tf.dtypes.float64) -> Grid:
    """Constructs a grid of the given dimension.
    Args:
        dim :int: the dimension of the grid
        boundary :Sequence: the boundaries of the grid
        grid_size :Sequence: the number of grid points in each direction
        dtype: the datatype of the grid points
    Returns:
        a grid of the given values
    """
    if dim == 1:
        g: Grid = get_grid(boundary, grid_size, dtype=dtype)

        return Grid(x=g.x)

    elif dim == 2:
        x: Grid = get_grid(boundary[0], grid_size[0], dtype=dtype)
        y: Grid = get_grid(boundary[1], grid_size[1], dtype=dtype)

        return Grid(x=x.x, y=y.x)


def rescale(grid: list[float], *, old_int: list[float], new_int: list[float]) -> list[float]:
    """Rescale a list of points to an interval"""
    return [new_int[0] + (new_int[1] - new_int[0]) * (g - old_int[0]) for g in grid]


def get_random_points(grid: Grid, *, n_points) -> Sequence[Union[Sequence, float]]:
    """Returns random points within a grid"""
    if grid.dim == 1:
        return rescale([p[0] for p in lhs(1, n_points)], old_int=[0, 1], new_int=grid.boundary)

    elif grid.dim == 2:
        pts_x: list = rescale([p[0] for p in lhs(1, n_points)], old_int=[0, 1], new_int=[grid.x[0], grid.x[-1]])
        pts_y: list = rescale([p[0] for p in lhs(1, n_points)], old_int=[0, 1], new_int=[grid.y[0], grid.y[-1]])

        return [[pts_x[i], pts_y[i]] for i in range(len(pts_x))]


def rescale_quads(quads: DataGrid, dim: int, *, domain: Sequence) -> DataGrid:
    """Rescales quadrature points on a grid to an arbitrary domain
    Args:
        quads: a DataGrid of quadrature points
        dim: the dimension of the domain
        domain: a (one or two)-dimensional rectangular domain, given by two diagonal corner points
    Returns:
        a DataGrid of quadrature points rescaled to the domain
    """

    if dim == 1:
        scaled_grid = [domain[0] + (p + 1) * (domain[1] - domain[0])/2 for p in quads.grid.x]

        return DataGrid(x=Grid(x=scaled_grid), f=quads.data)

    elif dim == 2:
        scaled_grid_x = []
        scaled_grid_y = []
        for p in quads.grid.x:
            scaled_grid_x.append(domain[0][0] + (p + 1) * (domain[0][1] - domain[0][0])/2)
        for p in quads.grid.y:
            scaled_grid_y.append(domain[1][0] + (p + 1) * (domain[1][1] - domain[1][0])/2)

        return DataGrid(x=Grid(x=scaled_grid_x, y=scaled_grid_y), f=quads.data)


def get_quadrature_data(dim, grid_size, *, dtype: tf.DType = tf.dtypes.float64) -> DataGrid:
    """Get one or two-dimensional quadrature points"""
    if dim == 1:
        return GaussLobattoJacobiWeights(grid_size, a=0, b=0, dtype=dtype)

    elif dim == 2:
        q_x: DataGrid = GaussLobattoJacobiWeights(grid_size[0], a=0, b=0, dtype=dtype)
        q_y: DataGrid = GaussLobattoJacobiWeights(grid_size[1], a=0, b=0, dtype=dtype)
        q_res = []
        for w_y in q_y.data:
            for w_x in q_x.data:
                q_res.append(w_x*w_y)

        return DataGrid(x=Grid(x=q_x.x, y=q_y.x), f=q_res)


def jacobian(points):
    """Return the volume of a hypercube defined by two diagonal points"""

    return np.prod([0.5 * (points[1][_] - points[0][_]) for _ in range(len(points[0]))])


def integrate_f(func, quads: DataGrid, test_func_vals, dim: int, *, domain) -> float:
    """Integrates a function against a test function on a rectangular domain,
    using quadrature"""

    j = jacobian(domain)
    quads_scaled: DataGrid = rescale_quads(quads, dim, domain=domain)
    res = np.sum(
        [j * func(quads_scaled.grid.data[i]) * test_func_vals[i] * quads.data[i]
         for i in range(quads.grid.size)]).item()

    return res


def integrate_f_over_grid(func, grid: Grid, quads: DataGrid, test_functions, dim: int) -> list[list[float]]:
    """Integrates a function over a grid for a set of test functions"""
    f_integrated: list[list[float]] = []
    if dim == 1:
        for i in range(len(grid.x) - 1):
            grid_element = [grid.x[i], grid.x[i + 1]]
            f_integrated.append(
                [integrate_f(func, quads, t, dim, domain=grid_element) for t in test_functions]
            )
    elif dim == 2:
        for j in range(len(grid.y) - 1):
            for i in range(len(grid.x) - 1):
                grid_element = [[grid.x[i], grid.x[i+1]], [grid.y[j], grid.y[j + 1]]]
                f_integrated.append(
                    [integrate_f(func, quads, t, dim, domain=grid_element) for t in test_functions]
                )

    return f_integrated
