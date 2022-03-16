import torch

from Code.Utils.utils import integrate

"""Variational loss for the one-dimensional weak PDE equation"""


def Weak1D(u,
           grid,
           f_integrated,
           d1test_func_vals,
           var_form):
    """Calculates the variational loss for the weak equation partial_x (u^3) = 1

    :param u: the neural network approximation
    :param grid: the domain of integration
    :param f_integrated: the values of the external function integrated against all test functions
    :param d1test_func_vals: derivatives of the test function
    :param var_form:
    :return:
    """

    # Track the variational loss
    loss_v = torch.tensor(0.0, requires_grad=True)

    # TO DO
    if var_form == 0:

        return torch.tensor([0.0])

    # Should output a warning if var_form > 1?
    else:

        u_3 = torch.zeros((len(grid.interior), 1), requires_grad=True)
        u_3 = u_3 + torch.pow(u(grid.interior), 3)
        for i in range(f_integrated.size):
            x = -1 * integrate(u_3, d1test_func_vals.data[i], grid.volume)
            x = x - f_integrated.data[i]
            x = torch.square(x.clone())
            loss_v = loss_v + x
            del x

    return loss_v / f_integrated.size
