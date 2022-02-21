import numpy as np
import tensorflow as tf
from typing import Sequence

from .data_types import DataGrid
from .functions import u as u_exact
from .test_functions import dtest_function, test_function
from .utils import integrate_over_grid

def integrate_Poisson(u, du, laplace, n_test_func, quads, quads_scaled, jacobians, grid_boundary, var_form,
                      dtype: tf.DType = tf.dtypes.float64) -> tf.Tensor:
    """Integrates variational forms of the Poisson equation over the grid against a particular test function,
    using quadrature.

    Args:
        u: the neural net itself. Returns a scalar when evaluated
        du: the gradient of the neural net. Returns a vector when evaluated.
        laplace: the Laplacian
        n_test_func: the number of the test function to use
        quads: the quadrature points
        quads_scaled: the quadrature points rescaled to the grid
        jacobians: the jacobians for each grid element
        grid_boundary: the grid boundary
        var_form: the variational form to use (1,2, or 3)
        dtype: the return value data type
    Returns:
        the variational integral
    """

    # Derivatives of the test functions for convenience
    def test_func(x):
        return test_function(x, n=n_test_func)

    def d1test_func(x):
        return dtest_function(x, n=n_test_func, d=1)

    def laplace_test_func(x):
        return np.sum(dtest_function(x, n=n_test_func, d=2))

    # Variational integral terms
    if var_form == 1:

        return tf.scalar_mul(-1, integrate_over_grid(laplace, test_func, quads, quads_scaled, jacobians,
                                                     as_tensor=True, dtype=dtype))
    elif var_form == 2:

        return integrate_over_grid(du, d1test_func, quads, quads_scaled, jacobians,
                                   as_tensor=True, dtype=dtype)

    elif var_form == 3:

        integral = integrate_over_grid(u, laplace_test_func, quads, quads_scaled, jacobians, as_tensor=True,
                                       dtype=dtype)
        boundary_term = np.sum([u_exact(grid_boundary[i]) * np.sum(d1test_func(grid_boundary[i]))
                                for i in range(len(grid_boundary))]).item()

        return tf.subtract(boundary_term, integral)


def integrate_Burger(u, du, laplace, n_test_func, quads, quads_scaled, jacobians, grid_boundary, var_form,
                     dtype: tf.DType = tf.dtypes.float64, nu: float = 0) -> tf.Tensor:
    """Integrates variational forms of the Burger equation over the grid against a particular test function,
    using quadrature.

    Args:
        u: the neural net itself. Returns a scalar when evaluated
        du: the gradient of the neural net. Returns a vector when evaluated.
        laplace: the Laplacian
        n_test_func: the number of the test function to use
        quads: the quadrature points
        quads_scaled: the quadrature points rescaled to the grid
        jacobians: the jacobians for each grid element
        grid_boundary: the grid boundary
        var_form: the variational form to use (1,2, or 3)
        dtype: the return value data type
        nu: the viscosity parameter of the Burger equation
    Returns:
        the variational integral
    """

    # Derivatives of the test functions for convenience
    def test_func(x):
        return test_function(x, n=n_test_func)

    def d1test_func(x):
        return dtest_function(x, n=n_test_func, d=1)

    def d2test_func(x):
        return dtest_function(x, n=n_test_func, d=2)

    # Variational integral terms
    if var_form == 1:

        return integrate_over_grid(
            lambda x: nu * laplace(x) - du(x)[(0, 1)] - u(x) * du(x)[(0, 0)], test_func,
            quads, quads_scaled, jacobians, as_tensor=True, dtype=dtype)

    elif var_form == 2:

        integral_1 = integrate_over_grid(
            lambda x: nu * du(x)[(0, 0)], lambda x: d1test_func(x)[0],
            quads, quads_scaled, jacobians, as_tensor=True, dtype=dtype)

        integral_2 = integrate_over_grid(
            lambda x: du(x)[(0, 1)] + u(x) * du(x)[(0, 0)], test_func, quads, quads_scaled, jacobians, as_tensor=True)

        return tf.scalar_mul(-1, tf.add(integral_1, integral_2))

    elif var_form == 3:

        integral_1 = integrate_over_grid(u, lambda x: d2test_func(x)[0],
                                         quads, quads_scaled, jacobians, as_tensor=True)
        integral_2 = integrate_over_grid(u, lambda x: d1test_func(x)[1],
                                         quads, quads_scaled, jacobians, as_tensor=True)
        integral_3 = integrate_over_grid(lambda x: u(x) ** 2, lambda x: d1test_func(x)[0],
                                         quads, quads_scaled, jacobians, as_tensor=True)
        boundary_term = np.sum([u_exact(grid_boundary[i]) * np.sum(d1test_func(grid_boundary[i]))
                                for i in range(len(grid_boundary))]).item()

        return tf.add(tf.add(tf.subtract(tf.scalar_mul(nu, integral_1), boundary_term), integral_2), integral_3)


def var_sum(*, u, du, ddu, n_test_func,
            quads: DataGrid, quads_scaled: Sequence[DataGrid], jacobians: Sequence[float], grid_boundary: Sequence,
            var_form: int, eq_type: str, pde_params: dict = None,
            dtype: tf.DType = tf.dtypes.float64) -> tf.Tensor:
    """Calculates the total variational form over a grid

    Args:
        u: the neural net
        du: the gradient of the neural net
        ddu: the second derivative of the neural net
        n_test_func: the number of the test function against which to integrate
        quads: the quadrature points
        quads_scaled: the quadrature points scaled to each grid element
        jacobians: the jacobians of the scaling transforms
        grid_boundary: the boundary of the grid
        var_form: which variational form to use
        eq_type: the type of differential equation used
        pde_params: constants used for the equations
        dtype: the data type of the return
    Returns:
        the variational form
    """

    # Return forms for various equations

    def laplace(x):
        return tf.reduce_sum(ddu(x))

    if eq_type == 'Poisson':

        return integrate_Poisson(u, du, laplace, n_test_func, quads, quads_scaled, jacobians, grid_boundary, var_form,
                                 dtype=dtype)

    elif eq_type == 'Helmholtz':

        k = 1 if pde_params is None else pde_params['Helmholtz']

        return tf.subtract(
            integrate_Poisson(u, du, laplace, quads, quads_scaled, jacobians, grid_boundary, var_form, dtype),
            integrate_over_grid(lambda x: k * u(x), lambda x: test_function(x, n=n_test_func), quads,
                                quads_scaled, jacobians, as_tensor=True, dtype=dtype))

    elif eq_type == 'Burger':
        nu = 0 if pde_params is None else pde_params['Burger']

        return integrate_Burger(u, du, laplace, n_test_func, quads, quads_scaled, jacobians, grid_boundary, var_form,
                                dtype=dtype, nu=nu)
