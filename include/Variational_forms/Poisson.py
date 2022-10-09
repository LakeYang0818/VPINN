import numpy as np
import torch

from ..integration import integrate

"""Variational loss for the Poisson equation"""


# TO DO: the second variational form does not work correctly.


def Poisson(
    u,
    du,
    ddu,
    grid,
    grid_boundary,
    normals,
    f_integrated,
    test_func_vals,
    weights,
    d1test_func_vals=None,
    d2test_func_vals=None,
    d1test_func_vals_bd=None,
    var_form=0,
):
    """Calculates the variational loss for the Poisson equation.

    :param u: the model itself
    :param du: its first derivative
    :param ddu: its second derivative
    :param grid: the grid over which to integrate
    :param f_integrated: the values of the external function integrated against every test function over the grid
    :param test_func_vals: the values of the test functions on the grid interior
    :param weights: the weights used for each test function
    :param d1test_func_vals: their first derivatives
    :param d2test_func_vals: their second derivatives
    :param d1test_func_vals_bd: the test functions' first derivatives evaluated on the grid boundary
    :param var_form: the variational form to use
    :return: the variational loss
    """

    # Track the variational loss
    loss_v = torch.tensor(0.0, requires_grad=True)

    if var_form == 0:

        laplace = torch.sum(ddu(grid, requires_grad=True), dim=-1, keepdim=True)

        for idx in range(len(f_integrated.coords["tf_idx"])):

            tf_vals = torch.reshape(
                torch.from_numpy(test_func_vals.isel(tf_idx=[idx]).data).float(),
                (-1, 1),
            )

            loss_v = loss_v + torch.square(
                integrate(
                    laplace,
                    tf_vals,
                    domain_density=test_func_vals.attrs["grid_density"],
                )
                - torch.from_numpy(f_integrated.isel(tf_idx=idx).data)
            ) * (weights[idx] ** 2)

    elif var_form == 1:

        grad = du(grid, requires_grad=True)

        for idx in range(len(f_integrated.coords["tf_idx"])):

            tf_vals = torch.reshape(
                torch.from_numpy(d1test_func_vals.isel(tf_idx=[idx]).data).float(),
                (-1, 1),
            )

            loss_v = loss_v + torch.square(
                integrate(
                    grad,
                    tf_vals,
                    domain_density=test_func_vals.attrs["grid_density"],
                )
                + torch.from_numpy(f_integrated.isel(tf_idx=idx).data)
            ) * (weights[idx] ** 2)

    elif var_form == 2:

        u_bd = u(grid_boundary)
        u_int = u(grid)

        for idx in range(len(f_integrated.coords["tf_idx"])):

            tf_vals = torch.sum(
                torch.reshape(
                    torch.from_numpy(d2test_func_vals.isel(tf_idx=[idx]).data).float(),
                    (-1, 1),
                ),
                dim=1,
                keepdim=True,
            )

            tf_vals_bd = torch.mul(
                torch.reshape(
                    torch.from_numpy(
                        d1test_func_vals_bd.isel(tf_idx=[idx]).data
                    ).float(),
                    (-1, 1),
                ),
                normals,
            )

            loss_v = loss_v + torch.square(
                integrate(
                    u_int,
                    tf_vals,
                    domain_density=test_func_vals.attrs["grid_density"],
                )
                - integrate(
                    u_bd,
                    torch.sum(
                        tf_vals_bd,
                        dim=1,
                        keepdim=True,
                    ),
                    domain_density=test_func_vals.attrs["grid_density"],
                )
                - f_integrated.data[idx]
            ) * (weights[idx] ** 2)

    return loss_v / len(f_integrated.coords["tf_idx"])
