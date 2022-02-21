from typing import Sequence, Union
import tensorflow as tf
import numpy as np
from pyDOE import lhs

from .data_types import DataGrid, Grid
from .test_functions import GaussLobattoJacobiWeights, test_function


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
    step_size = (1.0 * upper - lower) / (1.0 * grid_size - 1)
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
        a Grid of the given values
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
                q_res.append(w_x * w_y)

        return DataGrid(x=Grid(x=q_x.x, y=q_y.x), f=q_res)


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
        scaled_grid = [domain[0] + (p + 1) * (domain[1] - domain[0]) / 2 for p in quads.grid.x]

        return DataGrid(x=Grid(x=scaled_grid), f=quads.data)

    elif dim == 2:
        scaled_grid_x = []
        scaled_grid_y = []
        for p in quads.grid.x:
            scaled_grid_x.append(domain[0][0] + (p + 1) * (domain[0][1] - domain[0][0]) / 2)
        for p in quads.grid.y:
            scaled_grid_y.append(domain[1][0] + (p + 1) * (domain[1][1] - domain[1][0]) / 2)

        return DataGrid(x=Grid(x=scaled_grid_x, y=scaled_grid_y), f=quads.data)


def jacobian(points: Sequence) -> float:
    """Return the Jacobian of the coordinate transformation from a regular grid to the
    quadrature grid. For a 1d interval, this is simply given by half the length of the interval.
    For higher dimensional domains, it is the product of the Jacobians of their individual axes."""

    if isinstance(points[0], Sequence):
        return np.prod([0.5 * (points[1][_] - points[0][_]) for _ in range(len(points[0]))]).item()
    else:
        return 0.5 * (points[1] - points[0])


def scale_quadrature_data(grid: Grid, quads: DataGrid) -> (Sequence[DataGrid], Sequence[float]):
    quads_scaled = []
    jacobians = []
    if grid.dim == 1:
        for i in range(len(grid.x) - 1):
            element = [grid.x[i], grid.x[i + 1]]
            jac = jacobian(element)
            quads_rescaled = rescale_quads(quads, grid.dim, domain=element)

            quads_scaled.append(quads_rescaled)
            jacobians.append(jac)

    elif grid.dim == 2:
        for j in range(len(grid.y) - 1):
            for i in range(len(grid.x) - 1):
                element = [[grid.x[i], grid.x[i + 1]], [grid.y[j], grid.y[j + 1]]]
                jac = jacobian(element)
                quads_rescaled = rescale_quads(quads, grid.dim, domain=element)

                quads_scaled.append(quads_rescaled)
                jacobians.append(jac)

    return quads_scaled, jacobians


def integrate_over_grid(func, test_func, quads: DataGrid, quads_scaled: Sequence[DataGrid], jacobians: Sequence[float],
                        *, as_tensor: bool = False, dtype: tf.DType = tf.dtypes.float64):
    """Integrates a function against a test function over a grid using quadrature

    Args:
        func: the function to integrate
        test_func: the test function to use. Test function and function are multiplied using the dot product
        quads: the quadrature data over which to integrate
        quads: the quadrature data scaled to each element of the grid
        jacobians: the jacobians of the coordinate transforms
        as_tensor: whether to use tf.Tensors for the function input, and whether to return tf.Tensor types
        dtype: the data type to use for the tensors
    Returns:
        the value of the integral
    """
    if not as_tensor:
        res = 0
        for j in range(len(quads_scaled)):
            res += np.sum(
                [jacobians[j] * np.sum(func(quads_scaled[j].grid.data[i]) * test_func(quads.grid.data[i]), axis=None)
                 * quads.data[i]
                 for i in range(quads.grid.size)]).item()
        return res

    else:
        res = tf.constant([0], dtype=dtype)
        for j in range(len(quads_scaled)):
            val = tf.stack(
                     [tf.multiply(
                         tf.tensordot(func(tf.constant([quads_scaled[j].grid.data[i]], dtype=dtype)),
                                      tf.constant(test_func(quads.grid.data[i]), dtype=dtype), 0),
                         (jacobians[j] * quads.data[i]))
                      for i in range(quads.grid.size)
                      ])
            res = tf.add(res, tf.reduce_sum(val))
        return res
