import numpy as np
import torch

from Utils import integrate
from Utils.Datatypes.Grid import Grid

"""Tests the integration function"""


def constant_func(grid: Grid):
    if not grid.is_tensor:
        return np.array([[1.0] for _ in range(grid.size)])
    else:
        return torch.reshape(torch.ones(grid.size), (grid.size, 1))

def test_1d_integration():

    # Pairs of domains, functions, and resulting integral values
    to_test = [
        {
            'space': Grid(x=np.linspace(1, 6, 100), as_tensor=False, dtype=np.float64),
            'function': lambda x: constant_func(x),
            'result': 5
        },
        {
            'space': Grid(x=np.linspace(0, 2 * np.pi, 400), as_tensor=False, dtype=np.float64),
            'function': lambda x: np.sin(x.x),
            'result': 0.0
        },
        {
            'space': Grid(x=np.linspace(0, 2 * np.pi, 100000), as_tensor=False, dtype=np.float64),
            'function': lambda x: np.sin(x.x)**2,
            'result': np.pi
        },
        {
            'space': Grid(x=np.linspace(1, 4, 100), as_tensor=True),
            'function': lambda x: constant_func(x),
            'result': 3
        },
        {
            'space': Grid(x=np.linspace(-3, 3, 1000000), as_tensor=True, requires_grad=False),
            'function': lambda x: torch.pow(x.x, 2),
            'result': 18.0
        }
    ]

    for test_obj in to_test:
        integral = integrate(test_obj['function'](test_obj['space']),
                             domain_volume=test_obj['space'].volume,
                             as_tensor=test_obj['space'].is_tensor)
        if not test_obj['space'].is_tensor:
            np.testing.assert_allclose(integral, test_obj['result'], atol=1e-4)
        else:
            np.testing.assert_allclose(integral.detach().numpy(), test_obj['result'], atol=1e-4)

# Scalar functions on 2D domains
def test_2d_integration():
    def constant_func(grid: Grid):
        if not grid.is_tensor:
            return np.array([1.0 for _ in range(grid.size)])
        else:
            return torch.ones(grid.size)

    # Pairs of domains, functions, and resulting integral values
    to_test = [
        {
            'space': Grid(x=np.linspace(1, 6, 100), y=np.linspace(-2, 3, 85), as_tensor=False, dtype=np.float64),
            'function': lambda x: constant_func(x),
            'result': 25,
            'tolerance': 1e-5
        },
        {
            'space': Grid(x=np.linspace(0, np.pi, 500), y=np.linspace(0, np.pi, 500), as_tensor=False, dtype=np.float64),
            'function': lambda x: [np.sin(p[0]) * np.sin(p[1]) for p in x.data],
            'result': 4.0,
            'tolerance': 1e-1
        },
        {
            'space': Grid(x=np.linspace(0, 2 * np.pi, 500), y=np.linspace(0, np.pi, 500), as_tensor=True, requires_grad=True),
            'function': lambda x: torch.stack([torch.sin(p[0]) * torch.sin(p[1]) for p in x.data]),
            'result': 0.0,
            'tolerance': 1e-7
        }
    ]

    for test_obj in to_test:
        integral = integrate(test_obj['function'](test_obj['space']),
                             domain_volume=test_obj['space'].volume,
                             as_tensor=test_obj['space'].is_tensor)
        if not test_obj['space'].is_tensor:
            np.testing.assert_allclose(integral, test_obj['result'], atol=test_obj['tolerance'])
        else:
            np.testing.assert_allclose(integral.detach().numpy(), test_obj['result'], atol=test_obj['tolerance'])

def test_vector_integration():

    # Integrate x^2 + y^2 over a square domain
    grid1 = Grid(x=np.linspace(-1, 1, 600), y=np.linspace(-1, 1, 600), as_tensor=True)
    integral1 = integrate(grid1.data, grid1.data, domain_volume=4)

    np.testing.assert_allclose(integral1, 8/3*grid1.volume**2/(2**4), atol=1e-1)

    # Integrate sin(x)*sin(y) over a square domain
    grid2 = Grid(x=np.linspace(0, np.pi, 100), y=np.linspace(0, np.pi, 100), as_tensor=False, dtype=float)
    func_1 = [np.sin(x[0])*np.sin(x[1]) for x in grid2.data]
    integral2 = integrate(func_1, domain_volume=np.pi**2, as_tensor=False)

    np.testing.assert_allclose(integral2, 4, atol=1e-1)

