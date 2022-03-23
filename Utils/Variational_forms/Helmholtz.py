import torch

from Code.Utils.utils import integrate

"""Variational loss for the Helmholtz equation"""


def Helmholtz(u,
              du,
              ddu,
              grid,
              f_integrated,
              test_func_vals,
              d1test_func_vals,
              d2test_func_vals,
              d1test_func_vals_bd,
              var_form,
              pde_constants,
              weight_function = lambda x: 1):
    """Calculates the variational loss for the Helmholtz equation.

    :param u: the model itself
    :param du: its first derivative
    :param ddu: its second derivative
    :param grid: the grid over which to integrate
    :param f_integrated: the values of the external function integrated against every test function over the grid
    :param test_func_vals: the values of the test functions on the grid interior
    :param d1test_func_vals: their first derivatives
    :param d2test_func_vals: their second derivatives
    :param d1test_func_vals_bd: the test functions' first derivatives evaluated on the grid boundary
    :param var_form: the variational form to use
    :param pde_constants: the constants for the equation
    :param weight_function: the weighting function for the contributions of the individual test functions
    :return: the variational loss
    """

    # Track the variational loss
    loss_v = torch.tensor(0.0, requires_grad=True)

    k = pde_constants['Helmholtz']

    if var_form == 0:

        laplace = torch.sum(ddu(grid.interior, requires_grad=True), dim=1, keepdim=True)

        for i in range(f_integrated.size):
            loss_v = loss_v + torch.square(
                integrate(laplace, test_func_vals.data[i], domain_volume=grid.volume)
                + k * integrate(u(grid.interior), test_func_vals.data[i], domain_volume=grid.volume)
                - f_integrated.data[i]
            ) * weight_function(test_func_vals.coords[i])

    elif var_form == 1:

        grad = du(grid.interior, requires_grad=True)

        for i in range(f_integrated.size):
            loss_v = loss_v + torch.square(
                - integrate(grad, d1test_func_vals.data[i], domain_volume=grid.volume)
                + k * integrate(u(grid.interior), test_func_vals.data[i], domain_volume=grid.volume)
                - f_integrated.data[i]
            ) * weight_function(test_func_vals.coords[i])

    elif var_form == 2:

        u_bd = u(grid.boundary)
        u_int = u(grid.interior)

        for i in range(f_integrated.size):
            loss_v = loss_v + torch.square(
                integrate(u_int, torch.sum(d2test_func_vals.data[i], dim=1, keepdim=True), domain_volume=grid.volume)
                - integrate(u_bd, torch.sum(torch.mul(d1test_func_vals_bd.data[i], grid.normals), dim=1, keepdim=True),
                            domain_volume=grid.volume)
                + k * integrate(u(grid.interior), test_func_vals.data[i], domain_volume=grid.volume)
                - f_integrated.data[i]
            ) * weight_function(test_func_vals.coords[i])

    return loss_v / f_integrated.size
