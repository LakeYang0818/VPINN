"""Variational residuals"""
import tensorflow as tf
from .data_types import DataSet, Grid
from .test_functions import test_function, dtest_function

def U_NN_Poisson_1d(var_form: int, quad_data: DataSet, n_test_func: int, grid: Grid, boundary: list) -> object:

    """Calculates the variational forms for the Poisson equation on a one-dimensional grid.
    Args:
        var_form: the variational form. Can be 1, 2, or 3
        quad_data: the quadrature data
        n_test_func: which test function to use
    Returns:
        a tf.Tensor object
    """
    # Evaluate the test function on the quadrature points
    test_quad_element = test_function(n_test_func, quad_data.coords)

    d1_test_quad_element, d2_test_quad_element = dtest_function(Ntest_element, quad_data.x)

    d1test_bound_element, d2test_bound_element = dtest_function(Ntest_element, boundary)

    res = []
    if var_form == 1:
        for i in range(Ntest_element):
            jacobian = (grid[e + 1] - grid[e]) / 2
            p = quad_data.y*d2u_NN_quad_element*test_quad_element[i]
            p = tf.reduce_sum(p)
            res.append(-1*jacobian*p)

    elif var_form == 2:
        for i in range(Ntest_element):
            p = quad_data.y*d1u_NN_quad_element*d1test_quad_element[i]
            p += quad_data.y*u_NN_quad_element*test_quad_element[i]
            p = tf.reduce_sum(p)
            res.append(p)
    else:
        for i in range(Ntest_element):
            p = quad_data.y*u_NN_quad_element*d2test_quad_element[i]
            p = tf.reduce_sum(p)
            q = u_NN_bound_element*np.array([-d1test_bound_element[i][0], d1test_bound_element[i][-1]])
            q = tf.reduce_sum(q)
            res.append(-1/jacobian * p + 1/jacobian * q)

    return tf.reshape(tf.stack(res), (-1, 1))

# def U_NN_Poisson_2d(var_form: int, ..)

def U_NN_Helmholtz():

    if var_form == 1:
        for i in range(Ntest_element):
            p = -jacobian*tf.reduce_sum(quad_data.y*d2u_NN_quad_element*test_quad_element[i])
            p += jacobian*k*tf.reduce_sum(quad_data.y*u_NN_quad_element*test_quad_element[i])

        res.append(p)

def U_NN(var_type, quad_data, Ntest_element, grid, boundary):
    if (var_type[1] == "Poisson1D"):
        return U_NN_Poisson_1d(var_type[0], quad_data, grid, boundary)
    elif (var_type[1] == "Poisson2d"):
        return U_NN_Poisson_2d(var_type[0], quad_data, grid, boundary)
    elif (var_type[1] == "Helmholtz"):
        return U_NN_Helmholtz(var_type[0], quad_data, grid, boundary)
