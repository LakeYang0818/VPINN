import torch

from ..integration import integrate

"""Variational loss for the one-dimensional weak PDE equation"""


def Weak1D(u,
           grid,
           f_integrated,
           d1test_func_vals,
           var_form,
           weight_function=lambda x: 1):
    """Calculates the variational loss for the weak equation partial_x (u^3) = 1

    :param u: the neural network approximation
    :param grid: the domain of integration
    :param f_integrated: the values of the external function integrated against all test functions
    :param d1test_func_vals: derivatives of the test function
    :param var_form: the variational form to use
    :param weight_function: the test function weight function
    :return:
    """

    # Track the variational loss
    loss_v = torch.tensor(0.0, requires_grad=True)

    if var_form == 1:

        u_3 = torch.pow(u(grid.interior), 3)

        for i in range(f_integrated.size):
            loss_v = loss_v + torch.square(
                integrate(u_3, d1test_func_vals.data[i], domain_volume=grid.volume)
                + f_integrated.data[i]
            ) * weight_function(d1test_func_vals.coords[i])

    # Other forms are not used for this equation
    else:

        return torch.tensor([0.0])

    return loss_v / f_integrated.size
