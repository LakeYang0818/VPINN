import torch

from ..integration import integrate

"""Variational loss for the 2D Burger's equation"""


def Burger(
    u, du, grid, f_integrated, test_func_vals, d1test_func_vals, var_form, pde_constants
):

    # Track the variational loss
    loss_v = torch.tensor(0.0, requires_grad=True)

    nu = pde_constants["Burger"]

    if var_form == 1:

        u = u(grid)
        u_vec = torch.reshape(
            torch.stack([0.5 * torch.pow(u, 2), u], dim=1), (len(u), 2)
        )
        du_x = torch.swapaxes(du(u, requires_grad=True), 0, 1)[0] if nu != 0 else None

        for idx in range(len(f_integrated.coords["tf_idx"])):

            tf_vals = torch.reshape(
                torch.from_numpy(d1test_func_vals.isel(tf_idx=[idx]).data).float(),
                (-1, 1),
            )

            if nu != 0:
                loss_v = loss_v + torch.square(
                    integrate(
                        u_vec,
                        tf_vals,
                        domain_density=test_func_vals.attrs["grid_density"],
                    )
                    - nu
                    * integrate(
                        du_x,
                        torch.transpose(tf_vals, 0, 1)[0],
                        domain_density=test_func_vals.attrs["grid_density"],
                    )
                )
            else:
                loss_v = loss_v + torch.square(
                    integrate(
                        u_vec,
                        tf_vals,
                        domain_density=test_func_vals.attrs["grid_density"],
                    )
                )

    return loss_v / f_integrated.size
