import torch

from ..integration import integrate

"""Variational loss for the 2D Burger's equation"""


def Burger(u,
           du,
           grid,
           f_integrated,
           test_func_vals,
           d1test_func_vals,
           var_form,
           pde_constants):

    # Track the variational loss
    loss_v = torch.tensor(0.0, requires_grad=True)

    nu = pde_constants['Burger']

    if var_form == 1:

        u = u(grid.interior)
        u_vec = torch.reshape(torch.stack([0.5 * torch.pow(u, 2), u], dim=1), (len(u), 2))
        du_x = torch.swapaxes(du(grid.interior, requires_grad=True), 0, 1)[0] if nu != 0 else None

        for i in range(f_integrated.size):
            if nu != 0:
                loss_v = loss_v + torch.square(
                    integrate(u_vec, d1test_func_vals.data[i], domain_volume=grid.volume)
                    - nu * integrate(du_x, torch.transpose(d1test_func_vals.data[i], 0, 1)[0], domain_volume=grid.volume)
                )
            else:
                loss_v = loss_v + torch.square(
                    integrate(u_vec, d1test_func_vals.data[i], domain_volume=grid.volume)
                )

    return loss_v / f_integrated.size
