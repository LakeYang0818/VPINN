import torch

from ..integration import integrate

"""Variational loss for the porous medium equation"""

# TO DO: Test function operations should be carried out before the training loop


def PorousMedium(
    u,
    du,
    grid,
    f_integrated,
    test_func_vals,
    d1test_func_vals,
    d2test_func_vals,
    d1test_func_vals_bd,
    var_form,
    pde_constants,
    weight_function=lambda x: 1,
):

    """Calculates the variational loss for the porous medium equation.

    :param u: the model itself
    :param du: its first derivative
    :param grid: the grid over which to integrate
    :param f_integrated: the values of the external function integrated against every test function over the grid
    :param test_func_vals: the values of the test functions on the grid interior
    :param d1test_func_vals: their first derivatives
    :param d2test_func_vals: their second derivatives
    :param d1test_func_vals_bd: the test functions' first derivatives evaluated on the grid boundary
    :param var_form: the variational form to use
    :param pde_constants: the constants for the equation
    :param weight_function: the weighting function for the contributions of the individual test functions
    :return: the variational loss
    """

    # Track the variational loss
    loss_v = torch.tensor(0.0, requires_grad=True)

    m = pde_constants["PorousMedium"]
    u_m = torch.pow(u(grid.interior), m)
    u_0 = u(grid.interior)

    if var_form == 0:

        # Time derivative dt_u
        dt_u = torch.reshape(
            torch.swapaxes(du(grid.interior, requires_grad=True), 0, 1)[1],
            (len(grid.interior), 1),
        )

        # Calculate the Laplacian ∆(u^m)
        _ = torch.autograd.grad(
            u_m, grid.interior, grad_outputs=torch.ones_like(u_m), create_graph=True
        )[0]
        ddu_m = torch.sum(
            torch.autograd.grad(
                _, grid.interior, grad_outputs=torch.ones_like(_), create_graph=True
            )[0],
            dim=1,
            keepdim=True,
        )

        for i in range(f_integrated.size):
            loss_v = loss_v + torch.square(
                integrate(dt_u, test_func_vals.data[i], domain_volume=grid.volume)
                - integrate(ddu_m, test_func_vals.data[i], domain_volume=grid.volume)
            ) * weight_function(test_func_vals.coords[i])

    elif var_form == 1:

        # x-derivative of u^m
        dx_u_m = torch.reshape(
            torch.transpose(
                torch.autograd.grad(
                    u_m,
                    grid.interior,
                    grad_outputs=torch.ones_like(u_m),
                    create_graph=True,
                )[0],
                0,
                1,
            )[0],
            (len(grid.interior), 1),
        )

        for i in range(f_integrated.size):

            # Get the test function derivative values for x and t only.
            # TO DO: This should happen before the training loop
            dx_v = torch.reshape(
                torch.transpose(d1test_func_vals.data[i], 0, 1)[0],
                (len(grid.interior), 1),
            )
            dt_v = torch.reshape(
                torch.transpose(d1test_func_vals.data[i], 0, 1)[1],
                (len(grid.interior), 1),
            )

            loss_v = loss_v + torch.square(
                integrate(u_0, dt_v, domain_volume=grid.volume)
                - integrate(dx_u_m, dx_v, domain_volume=grid.volume)
            ) * weight_function(d1test_func_vals.coords[i])

    elif var_form == 2:

        # Evaluate u^m on the right and left boundaries of the grid (x = +/- L)
        u_m_right = torch.pow(u(grid.right_boundary), m)
        u_m_left = torch.pow(u(grid.left_boundary), m)

        for i in range(f_integrated.size):
            loss_v = loss_v + torch.square(
                # Integrate u^m(L, t) on the right boundary [L, t]
                -integrate(
                    u_m_right,
                    # This is the d1test only on the right boundary
                    # TO DO: This should be a separate function or
                    # evaluated before the training loop
                    torch.reshape(
                        torch.transpose(
                            d1test_func_vals_bd.data[i][
                                len(grid.x) - 1 : len(grid.x) + len(grid.y) - 1
                            ],
                            0,
                            1,
                        )[0],
                        (len(grid.y), 1),
                    ),
                    domain_volume=grid.boundary_volume / 4,
                )
                # Integrate u^m(-L, t) on the left boundary [-L, t]
                + integrate(
                    u_m_left,
                    # This is the d1test only on the left boundary
                    # TO DO: should be a separate function or
                    # evaluated before the training loop
                    torch.reshape(
                        torch.cat(
                            (
                                torch.swapaxes(
                                    d1test_func_vals_bd.data[i][
                                        2 * len(grid.x) + len(grid.y) - 3 :
                                    ],
                                    0,
                                    1,
                                )[0],
                                torch.reshape(d1test_func_vals_bd.data[i][0][0], (1,)),
                            ),
                            dim=0,
                        ),
                        (len(grid.y), 1),
                    ),
                    domain_volume=grid.boundary_volume / 4,
                )
                # Integrate u^m against the second spacial derivative of the test function
                + integrate(
                    u_m,
                    torch.reshape(
                        torch.transpose(d2test_func_vals.data[i], 0, 1)[0],
                        (len(u_m), 1),
                    ),
                    domain_volume=grid.volume,
                )
                # Integrate u against the first time derivative of the test function
                + integrate(
                    u_0,
                    torch.reshape(
                        torch.transpose(d1test_func_vals.data[i], 0, 1)[1],
                        (len(u_m), 1),
                    ),
                    domain_volume=grid.volume,
                )
            ) * weight_function(test_func_vals.coords[i])

    return loss_v / f_integrated.size
