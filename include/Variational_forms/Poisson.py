import torch

from ..integration import integrate

"""Variational loss for the Poisson equation"""


def Poisson(
    variational_form: int,
    u: callable,
    du: callable,
    ddu: callable,
    grid: torch.Tensor,
    grid_density: torch.Tensor,
    f_integrated: torch.Tensor,
    test_func_vals: torch.Tensor,
    tf_weights: torch.Tensor,
    grid_boundary: torch.Tensor = None,
    normals: torch.Tensor = None,
    d1test_func_vals: torch.Tensor = None,
    d2test_func_vals: torch.Tensor = None,
    d1test_func_vals_bd: torch.Tensor = None,
) -> torch.Tensor:

    """Calculates the variational loss for the Poisson equation.

    :param variational_form: the variational form to use
    :param u: the neural network itself
    :param du: its first derivative
    :param ddu: its second derivative
    :param grid: the grid over which to integrate
    :param grid_density: the grid density
    :param f_integrated: the values of the external function integrated against every test function over the grid
    :param test_func_vals: the values of the test functions on the grid interior
    :param tf_weights: the weights used for each test function
    :param grid_boundary: the boundary of the grid
    :param normals: the normals of the grid boundary
    :param d1test_func_vals: the test functions' first derivatives
    :param d2test_func_vals: their second derivatives
    :param d1test_func_vals_bd: the test functions' first derivatives evaluated on the grid boundary
    :return: the variational loss
    """

    # Track the variational loss
    loss_v = torch.tensor(0.0, requires_grad=True)

    if variational_form == 0:

        laplace = torch.sum(ddu(grid, requires_grad=True), dim=-1, keepdim=True)

        for _ in range(len(f_integrated)):

            # Integrate the Laplacian against all the test functions
            integral = integrate(
                laplace,
                test_func_vals[_],
                domain_density=grid_density,
            )

            loss_v = loss_v + torch.square(integral - f_integrated[_]) * tf_weights[_]

    elif variational_form == 1:

        grad = du(grid, requires_grad=True)

        for _ in range(len(f_integrated)):

            # Integrate the Jacobian against all test functions
            integral = integrate(
                grad,
                d1test_func_vals[_],
                domain_density=grid_density,
            )
            loss_v = loss_v + torch.square(integral + f_integrated[_]) * tf_weights[_]

    elif variational_form == 2:

        u_bd = u(grid_boundary)
        u_int = u(grid)

        for _ in range(len(f_integrated)):

            loss_v = (
                loss_v
                + torch.square(
                    integrate(
                        u_int,
                        torch.sum(d2test_func_vals[_], dim=1, keepdim=True),
                        domain_density=grid_density,
                    )
                    - integrate(
                        u_bd,
                        torch.sum(
                            torch.mul(d1test_func_vals_bd[_], normals),
                            dim=1,
                            keepdim=True,
                        ),
                        domain_density=grid_density,
                    )
                    - f_integrated[_]
                )
                * tf_weights[_]
            )

    return loss_v / len(tf_weights)
