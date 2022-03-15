import torch

from Code_torch.Utils.utils import integrate

"""Variational loss for the Poisson equation"""


# TO DO: the second variational form does not work correctly.

def Poisson(u,
            du,
            ddu,
            grid,
            f_integrated,
            test_func_vals,
            d1test_func_vals,
            d2test_func_vals,
            d1test_func_vals_bd,
            var_form):
    """Calculates the variational loss for the Poisson equation.

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
    :return: the variational loss
    """

    # Track the variational loss
    loss_v = torch.tensor(0.0, requires_grad=True)

    if var_form == 0:

        laplace = torch.sum(ddu(grid.interior, requires_grad=True), dim=1, keepdim=True)

        for i in range(f_integrated.size):
            loss_v = loss_v + torch.square(
                integrate(laplace, test_func_vals.data[i], grid.volume) - f_integrated.data[i])*2**(-1.0*i)

    elif var_form == 1:

        grad = du(grid.interior, requires_grad=True)

        for i in range(f_integrated.size):
            loss_v = loss_v + torch.square(
                integrate(grad, d1test_func_vals.data[i], grid.volume) + f_integrated.data[i])*2**(-1.0*i)

    elif var_form == 2:

        u_bd = u(grid.boundary)
        u_int = u(grid.interior)

        # Need to multiply with normal vector of boundary?
        for i in range(f_integrated.size):
            loss_v = loss_v + torch.square(
                integrate(u_int, torch.sum(d2test_func_vals.data[i], dim=1, keepdim=True), grid.volume)
                - u_bd[-1]*d1test_func_vals_bd.data[i][-1] + u_bd[0]*d1test_func_vals_bd.data[i][0]
                - f_integrated.data[i])*2**(-1.0*i)

    return loss_v / f_integrated.size
