import torch

from ..integration import integrate

"""Variational loss for the 2D Burger's equation"""


def Burger(
    u: callable,
    du: callable,
    grid: torch.Tensor,
    grid_density: torch.Tensor,
    f_integrated: torch.Tensor,
    d1test_func_vals: torch.Tensor,
    tf_weights: torch.Tensor,
    nu: float,
) -> torch.Tensor:

    """Calculates the variational loss for Burger's equation.

    :param u: the neural network itself
    :param du: its first derivative
    :param grid: the grid over which to integrate
    :param grid_density: the grid density
    :param f_integrated: the values of the external function integrated against every test function over the grid
    :param d1test_func_vals: the values of the test function derivatives on the grid interior
    :param tf_weights: the weights used for each test function
    :param grid_boundary: the boundary of the grid
    :param normals: the normals of the grid boundary
    :param d1test_func_vals: the test functions' first derivatives
    :return: the variational loss
    """

    # Track the variational loss
    loss_v = torch.tensor(0.0, requires_grad=True)

    # Evaluate the neural net on the grid
    u = u(grid)

    # Calculate partial derivatives
    u_vec = torch.reshape(torch.stack([0.5 * torch.pow(u, 2), u], dim=1), (len(u), 2))
    du_x = torch.swapaxes(du(u, requires_grad=True), 0, 1)[0] if nu != 0 else None

    if nu != 0:

        integrals = torch.stack(
            [
                integrate(
                    u_vec,
                    d1test_func_vals[_],
                    domain_density=grid_density,
                )
                - nu
                * integrate(
                    du_x,
                    torch.transpose(d1test_func_vals[_], 0, 1)[0],
                    domain_density=grid_density,
                )
                for _ in range(len(d1test_func_vals))
            ]
        )

    else:
        integrals = torch.stack(
            [
                integrate(
                    u_vec,
                    d1test_func_vals[_],
                    domain_density=grid_density,
                )
                for _ in range(len(d1test_func_vals))
            ]
        )

    # Multiply each integral with the weight accorded each test function and calculate the mean squared error
    # For the first variational form, there is a sign flip in the equation
    loss_v = loss_v + torch.nn.functional.mse_loss(
        integrals * tf_weights, -1 * f_integrated * tf_weights
    )

    return loss_v
