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
              var_form,
              pde_constants):

    # Track the variational loss
    loss_v = torch.tensor(0.0, requires_grad=True)

    k = pde_constants['Helmholtz']

    if var_form == 0:

        laplace = torch.sum(ddu(grid.interior, requires_grad=True), dim=1, keepdim=True)
        for i in range(f_integrated.size):
            q = integrate(laplace, test_func_vals.data[i], domain_volume=grid.volume)
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
            q = q + k * integrate(u(grid.interior), test_func_vals.data[i], domain_volume=grid.volume)
            q = q - f_integrated.data[i]
            q = torch.square(q.clone())
            loss_v = loss_v + q
            del q

    return loss_v / f_integrated.size
