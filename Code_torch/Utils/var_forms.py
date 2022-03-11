import torch

# Local imports
from .Datatypes.Grid import Grid
from .utils import integrate

# TO DO: Add unit tests

def calculate_var_loss(u, du, ddu, grid: Grid, f_integrated,
                       test_func_vals, d1test_func_vals, d2_test_func_vals, d1test_func_vals_bd,
                       var_form,
                       eq_type,
                       pde_constants):
    """ Calculates the variational loss for different equation types.

    :param u:
    :param du:
    :param ddu:
    :param grid:
    :param f_integrated:
    :param test_func_vals:
    :param d1test_func_vals:
    :param d2_test_func_vals:
    :param d1test_func_vals_bd:
    :param var_form:
    :param eq_type:
    :param pde_constants:
    :return:
    """

    loss_v = torch.tensor(0.0, requires_grad=True)

    if eq_type == "Poisson":

        if var_form == 0:

            laplace = torch.sum(ddu(grid.interior, requires_grad=True), dim=1, keepdim=True)
            for i in range(f_integrated.size):
                q = integrate(laplace, test_func_vals.data[i], grid.volume) - f_integrated.data[i]
                q = torch.square(q.clone())
                loss_v = loss_v + q
                del q

        elif var_form == 1:

            integral_norm = grid.volume / len(test_func_vals.data[0])

            grad = du(grid.interior, requires_grad=True)

            for i in range(f_integrated.size):
                q = (-integral_norm * torch.einsum('ij, ij->', grad, d1test_func_vals.data[i]) - f_integrated.data[i])
                q = torch.square(q.clone())
                loss_v = loss_v + q
                del q

        # TO DO
        elif var_form == 2:

            return torch.tensor([0.0])

    elif eq_type == "Helmholtz":

        k = pde_constants['Helmholtz']

        if var_form == 0:

            laplace = torch.sum(ddu(grid.interior, requires_grad=True), dim=1, keepdim=True)
            for i in range(f_integrated.size):
                q = integrate(laplace, test_func_vals.data[i], grid.volume)
                q = q + k * integrate(u(grid.interior), test_func_vals.data[i], grid.volume)
                q = q - f_integrated.data[i]
                q = torch.square(q.clone())
                loss_v = loss_v + q
                del q

        elif var_form == 1:

            integral_norm = grid.volume / len(test_func_vals.data[0])

            grad = du(grid.interior, requires_grad=True)

            for i in range(f_integrated.size):
                q = -integral_norm * torch.einsum('ij, ij->', grad, d1test_func_vals.data[i])
                q = q + k * integrate(u(grid.interior), test_func_vals.data[i], grid.volume)
                q = q - f_integrated.data[i]
                q = torch.square(q.clone())
                loss_v = loss_v + q
                del q

    elif eq_type == "Weak1D":

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

    # ..... Everything below this point has not yet been properly checked ..............................................

    elif eq_type == 'Burger':

        nu = pde_constants['Burger']

        if var_form == 1:
            u = u(grid.interior)

            integral_norm = grid.volume / len(test_func_vals.data[0])

            u_vec = torch.reshape(torch.stack([0.5 * torch.pow(u, 2), u], dim=1), (len(u), 2))

            du_x = torch.swapaxes(du(grid.interior, requires_grad=True), 0, 1)[0] if nu > 0 else None

            for i in range(f_integrated.size):
                q = (integral_norm * torch.einsum('ij, ij->', u_vec, d1test_func_vals.data[i]))
                if nu > 0:
                    q = q.clone() - nu * integral_norm * torch.einsum('i, i->', du_x,
                                                          torch.swapaxes(d1test_func_vals.data[i], 0, 1)[0])
                q = q.clone() - f_integrated.data[i]
                q = torch.square(q.clone())
                loss_v = loss_v + q
                del q

    elif eq_type == 'PorousMedium':

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
                d_xx_v = torch.reshape(torch.swapaxes(d2_test_func_vals[i], 0, 1)[0], (len(d2_test_func_vals[i]), 1))

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
