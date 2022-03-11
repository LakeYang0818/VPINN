import numpy as np
import torch

from Utils import integrate
from Utils.Datatypes.Grid import construct_grid, Grid, rescale_grid

"""Tests the integration function"""


def test_1d_integration():

    def constant_func(grid: Grid):
        if not grid.is_tensor:
            return np.array([1.0 for _ in range(grid.size)])
        else:
            return torch.ones(grid.size)

    # Pairs of domains, functions, and resulting integral values
    to_test = [
        (Grid(x=np.linspace(1, 6, 100), as_tensor=False, dtype=np.float64), lambda x: constant_func(x), 5),
        (Grid(x=np.linspace(0, 2 * np.pi, 400), as_tensor=False, dtype=np.float64), lambda x: np.sin(x.x),  0.0),
        (Grid(x=np.linspace(0, 2 * np.pi, 100000), as_tensor=False, dtype=np.float64), lambda x: np.sin(x.x)**2, np.pi),
        (Grid(x=np.linspace(1, 4, 100), as_tensor=True), lambda x: constant_func(x), 3),
        (Grid(x=np.linspace(-3, 3, 1000000), as_tensor=True, requires_grad=False),
         lambda x: torch.flatten(torch.pow(x.x, 2)), 18.0)
    ]

    for test_obj in to_test:
        integral = integrate(test_obj[1](test_obj[0]), constant_func(test_obj[0]), test_obj[0].volume,
                             as_tensor=test_obj[0].is_tensor)
        if not test_obj[0].is_tensor:
            np.testing.assert_allclose(integral, test_obj[2], atol=1e-4)
        else:
            np.testing.assert_allclose(integral.detach().numpy(), test_obj[2], atol=1e-4)


def test_2d_integration():

    def constant_func(grid: Grid):
        if not grid.is_tensor:
            return np.array([1.0 for _ in range(grid.size)])
        else:
            return torch.ones(grid.size)

    # Pairs of domains, functions, and resulting integral values
    to_test = [
        (Grid(x=np.linspace(1, 6, 100), y=np.linspace(-2, 3, 85), as_tensor=False, dtype=np.float64),
            lambda x: constant_func(x), 25, 1e-5),
        (Grid(x=np.linspace(0, np.pi, 500), y=np.linspace(0, np.pi, 500), as_tensor=False, dtype=np.float64),
            lambda x: [np.sin(p[0])*np.sin(p[1]) for p in x.data], 4.0, 1e-1),
        (Grid(x=np.linspace(0, 2*np.pi, 500), y=np.linspace(0, np.pi, 500), as_tensor=True, requires_grad=True),
         lambda x: torch.stack([torch.sin(p[0]) * torch.sin(p[1]) for p in x.data]), 0.0, 1e-8)
    ]

    for test_obj in to_test:
        integral = integrate(test_obj[1](test_obj[0]), constant_func(test_obj[0]), test_obj[0].volume,
                             as_tensor=test_obj[0].is_tensor)
        if not test_obj[0].is_tensor:
            np.testing.assert_allclose(integral, test_obj[2], atol=test_obj[-1])
        else:
            np.testing.assert_allclose(integral.detach().numpy(), test_obj[2], atol=test_obj[-1])
