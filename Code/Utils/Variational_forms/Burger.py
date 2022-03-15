import torch

from Code_torch.Utils.utils import integrate

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

    return loss_v / f_integrated.size