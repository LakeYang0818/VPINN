import torch

from ..integration import integrate

"""Variational loss for the Poisson equation"""


def dummy(
    device: str,
    u: callable,
    grid: torch.Tensor,
    grid_density: torch.Tensor,
    f_integrated: torch.Tensor,
    test_func_vals: torch.Tensor,
    tf_weights: torch.Tensor,
) -> torch.Tensor:

    """Calculates the variational loss for the Poisson equation.

    :param device: the training device to use
    :param u: the neural network itself
    :param grid: the grid over which to integrate
    :param grid_density: the grid density
    :param f_integrated: the values of the external function integrated against every test function over the grid
    :param test_func_vals: the values of the test functions on the grid interior
    :param tf_weights: the weights used for each test function
    :return: the variational loss
    """

    # Track the variational loss
    loss_v = torch.tensor([0.0], requires_grad=True).to(device)

    u_int = u(grid)

    for _ in range(len(f_integrated)):

        # Integrate the Laplacian against all the test functions
        integral = integrate(
            u_int,
            test_func_vals[_],
            domain_density=grid_density,
        )

        loss_v = loss_v + torch.square(integral - f_integrated[_]) * tf_weights[_]

    return loss_v / len(tf_weights)
