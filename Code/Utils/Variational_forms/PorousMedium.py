import torch

from Code_torch.Utils.utils import integrate

"""Variational loss for the porous medium equation"""


# To do: this has not yet been checked!

def PorousMedium(u,
                 du,
                 ddu,
                 grid,
                 f_integrated,
                 test_func_vals,
                 d1test_func_vals,
                 d2test_func_vals,
                 d1test_func_vals_bd,
                 var_form,
                 pde_constants):
    """Calculates the variational loss for the porous medium equation.

    :param u:
    :param du:
    :param ddu:
    :param grid:
    :param f_integrated:
    :param test_func_vals:
    :param d1test_func_vals:
    :param d2test_func_vals:
    :param d1test_func_vals_bd:
    :param var_form:
    :param pde_constants:
    :return:
    """

    m = pde_constants['PorousMedium']
    u = u(grid.interior)
    u_m = torch.pow(u, m)

    integral_norm = grid.volume / len(test_func_vals.data[0])

    if var_form == 0:
        du = du(grid.interior, requires_grad=True)
        g = torch.autograd.grad(u_m, grid.interior, grad_outputs=torch.ones_like(u_m), create_graph=True)[0]
        gg = torch.autograd.grad(g, grid.interior, grad_outputs=torch.ones_like(g), create_graph=True)[0]

        for i in range(f_integrated.size):
            q = integrate(torch.swapaxes(du, 0, 1)[1], test_func_vals[i], grid.volume)
            q = q.clone() - integrate(gg, test_func_vals[i], grid.volume)
            q = torch.square(q)
            loss_v = loss_v + q
            del q

    elif var_form == 1:

        dx_u_m = torch.autograd.grad(u_m, grid.interior, grad_outputs=torch.ones_like(u_m), create_graph=True)[
            0]

        for i in range(f_integrated.size):
            q = -1 * integrate(u, torch.swapaxes(d1test_func_vals[i], 0, 1)[1], grid.volume)
            q = q.clone() + integral_norm * torch.einsum('i, i->', torch.swapaxes(dx_u_m, 0, 1)[1],
                                                         torch.swapaxes(d1test_func_vals[i], 0, 1)[0])
            q = q.clone() - f_integrated.data[i]
            q = torch.square(q.clone())
            loss_v = loss_v + q
            del q

    elif var_form == 2:

        for i in range(f_integrated.size):
            # second derivative of v in x direction
            d_xx_v = torch.reshape(torch.swapaxes(d2test_func_vals[i], 0, 1)[0], (len(d2test_func_vals[i]), 1))

            # first derivative of evaluated on x= {-L, +L}
            boundary_vals = d1test_func_vals_bd[i][len(grid.x):-len(grid.x)]
            t_boundary = grid.boundary[len(grid.x):-len(grid.x)]
            u_bd = u(t_boundary)
            u_bd = torch.pow(u_bd, 2)
            dx_u_2 = torch.autograd.grad(u_bd, t_boundary,
                                         grad_outputs=torch.ones_like(u_bd), create_graph=True)[0]

            q = - 1 * integrate(u, torch.swapaxes(d1test_func_vals[i], 0, 1)[1], grid.volume)
            q = q.clone() - integral_norm * torch.einsum('ij, ij->', u_m, d_xx_v)
            q = q.clone() + integral_norm * torch.einsum('i, i->', torch.swapaxes(dx_u_2, 0, 1)[0],
                                                         torch.swapaxes(boundary_vals, 0, 1)[0])
            q = q.clone() - f_integrated.data[i]
            q = torch.square(q.clone())
            loss_v = loss_v + q

    return loss_v / f_integrated.size
