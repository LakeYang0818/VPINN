from scipy.special import gamma
import numpy as np
import torch

from Utils import testfunc_grid_evaluation as tfunc_eval, test_function as tfunc, dtest_function
from Utils.Datatypes.Grid import construct_grid, Grid, rescale_grid

"""Tests the test functions"""


def check_test_function(type: str, index: int):
    # ... Test the output type for scalar outputs ......................................................................
    expect_scalar_outputs = [
        0.3,
        2,
        [-0.3, 0.4, -0.6],
        np.array([0.3, 0.6]),
        np.array([-1, 2., 0.4, 5]),
        torch.tensor(0.3, requires_grad=True),
        torch.tensor(4, dtype=torch.int),
        torch.tensor([0.4]),
        torch.tensor([0.2, 0.3, 0.4])
    ]

    for input in expect_scalar_outputs:
        val = tfunc(input, index, which=type)
        if isinstance(val, torch.Tensor):
            assert (val.dim() == 1)
            assert (val.size() == torch.Size([1]))
            assert (abs(val.detach().numpy()) >= 0)
            assert (isinstance(val, torch.Tensor))
        else:
            assert (np.shape(val) == ())
            assert (abs(val) >= 0)
            assert (not isinstance(val, torch.Tensor))

    # ... Test the output type for sequence outputs ....................................................................
    expect_sequence_outputs = [
        np.array([[0.3, 0.5], [1, 2]]),
        torch.stack([torch.tensor([0.3, 0.5]), torch.tensor([-1, 1])]),
        torch.tensor([[0, 1], [2, 3]]),
        [[-1, 1], [0.3, 0.4]],
        [[0], [1]],
        Grid(x=np.linspace(-1, 1, 11), as_tensor=False, dtype=np.float64),
        Grid(x=np.linspace(-1, 1, 11), as_tensor=True, requires_grad=False),
        Grid(x=np.linspace(-1, 1, 11), y=np.linspace(0, 1, 3), as_tensor=True, requires_grad=False),
    ]

    for input in expect_sequence_outputs:
        val = tfunc(input, index, which=type)
        if isinstance(input, Grid):
            assert np.shape(val) == (input.size, 1)
        else:
            assert np.shape(val) == (np.shape(input)[0], 1)


    # Check some explicit examples
    # Scalar output
    assert (tfunc(-1, index, which=type) == 0)
    assert (tfunc(torch.tensor([1]), index, which=type) == torch.tensor([0]))

    # Sequence output
    assert (tfunc([[-1, -1, -1], [-1, 1, 1]], index, which=type)[0] == 0)
    assert (tfunc([[-1, 1, -1], [-1, 1, 1]], index, which=type)[1] == 0)
    assert (tfunc(torch.tensor([-1, -1, -1]), index, which=type) == torch.tensor([0]))
    assert (tfunc(torch.tensor([[-1, -1, -1], [-1, 1, 1]]), index, which=type)[0] == torch.tensor([0]))
    assert (tfunc(torch.tensor([[-1, -1, -1], [-1, 1, 1]]), index, which=type)[1] == torch.tensor([0]))

    # ... Test the test function properties ............................................................................

    # Test the test functions vanish on the boundary
    grid_1d = construct_grid(dim=1, boundary=[-1, 1], grid_size=3)
    test_func_vals = tfunc(grid_1d.boundary, 2, which=type)
    assert (test_func_vals.numpy() == [0, 0]).all()

    grid_2d = construct_grid(dim=2, boundary=[[-1, 1], [-1, 1]], grid_size=[4, 4])
    test_func_vals = tfunc(grid_1d.boundary, 2, which=type)
    for val in test_func_vals:
        assert (val.numpy() == 0)


# ... Test the test function derivatives ...........................................................................
def check_test_function_derivatives(type: str, index: int, d: int):
    inputs = [
        [0.3],
        [0.3, 0.6],
        [[0.4, -2], [0.6, 0.8]],
        torch.tensor([0.3]),
        torch.tensor([0.4, 0.5, 0.7]),
        torch.tensor([[0.3, 0.1], [9., 1]]),
        torch.tensor([0.4, 0.5, 0.7], requires_grad=True),
        Grid(x=np.linspace(1, 2, 3), as_tensor=False, dtype=np.float64),
        Grid(x=np.linspace(1, 2, 3), as_tensor=True),
        Grid(x=np.linspace(1, 2, 3), y=np.linspace(0, 1, 3), as_tensor=False, dtype=np.float64),
        Grid(x=np.linspace(1, 2, 3), y=np.linspace(0, 1, 4), as_tensor=True)
    ]
    for input in inputs:
        vals = dtest_function(input, index, d=d, which=type)

        if isinstance(input, Grid):
            if d == 0:
                assert 1==1
            else:
                assert (input.size == len(vals))
        else:
            if d == 0:
                assert 1==1
            else:
                assert (np.shape(input) == np.shape(vals))

    test_array = [[-1], [0.3], [0.6], [1], [2]]

    # Test the zero-th derivative is equal to the function itself
    if d == 0:
        assert (dtest_function(test_array, index, d=d, which=type) == tfunc(test_array, index, which=type)).all()

    # The n-th Legendre/Chebyshev test function are polynomials of degree n+1, so the (d+2) derivative should vanish.
    # The d+1 derivative must be a constant.
    if type in ['Legendre', 'Chebyshev']:
        if d == index + 1:
            if type == 'Chebyshev':
                assert (dtest_function(test_array, index, d=d, which=type)
                        == np.ones(5) * gamma(index + 2) * 2 ** index).all()
            elif type == 'Legendre':
                assert (dtest_function(test_array, index, d=d, which=type)
                        == np.around(
                            np.ones(5) * gamma(index + 2) * 2 ** (index + 1) * gamma(index + 1.5) / (
                                    gamma(0.5) * gamma(index + 2)),
                            7)).all()
        if d > index + 1:
            assert (dtest_function(test_array, index, d=d, which=type) == [0, 0, 0, 0, 0]).all()


# Test function indices
indices = np.arange(1, 30, 1, dtype=int)
d = np.arange(0, 10, 1, dtype=int)

# Test the Legendre test functions
def test_legendre():
    for index in indices:
        check_test_function('Legendre', index)

# Test the Chebyshev test functions
def test_chebyshev():
    for index in indices:
        check_test_function('Chebyshev', index)


# Test the Legendre test function derivatives
def test_dlegendre():
    for index in indices:
        for order in d:
            check_test_function_derivatives('Legendre', index, order)


# Test the Chebyshev test function derivatives
def test_dchebyshev():
    for index in indices:
        for order in d:
            check_test_function_derivatives('Chebyshev', index, order)


# Test the evaluation of test functions on a grid
def test_evaluation_on_grid():
    domains = {'all': lambda x: x.data, 'boundary': lambda x: x.boundary, 'interior': lambda x: x.interior}

    # Test 1d case
    grids1d = [
        Grid(x=np.linspace(2, 4, 5), as_tensor=False, dtype=np.float64),
        Grid(x=np.linspace(2, 4, 7), as_tensor=True),
        Grid(x=np.linspace(2, 4, 9), as_tensor=True, requires_grad=True)
    ]
    for key, value in domains.items():
        for grid in grids1d:
            for order in d:
                vals = tfunc_eval(grid, 10, d=order, which='Legendre', where=key)

                # Test the output shape
                if not grid.is_tensor:
                    assert (vals.coords.flatten() == np.arange(1, 11, 1, dtype=float)).all()
                    assert len(np.shape(vals.data)) == 3
                    assert np.shape(vals.data) == (10, len(value(grid)), 1)
                else:
                    (vals.coords.detach().numpy() == np.arange(1, 11, 1, dtype=float)).all()
                    assert vals.data.dim() == 3
                    assert vals.data.shape == torch.Size([10, len(value(grid)), 1])

                # Test the function values
                grid_rescaled = rescale_grid(grid, new_domain=[-1, 1],
                                             as_tensor=grid.is_tensor, requires_grad=grid.requires_grad)
                for i in range(len(vals.data)):
                    assert (vals.data[i] == dtest_function(value(grid_rescaled), i + 1, d=order,
                                                           which='Legendre')).all()
                    if order == 0 and key != 'interior':
                        assert (vals.data[i][0] == 0)
                        assert (vals.data[i][-1] == 0)

                if grid.requires_grad:
                    assert vals.data[0][0].requires_grad == True

    # Test 2d case
    grids2d = [
        # Grid(x=np.linspace(2, 4, 5), y=np.linspace(1, 2, 4), as_tensor=False, dtype=float),  // this doesnt work: line 221 in test_functions.py
        Grid(x=np.linspace(2, 4, 5), y=np.linspace(-3, -1, 4), as_tensor=True),
        Grid(x=np.linspace(-1, 8, 5), y=np.linspace(-3, -1, 4), as_tensor=True, requires_grad=True)
    ]

    for key, value in domains.items():
        for grid in grids2d:
            for order in d:
                vals = tfunc_eval(grid, [4, 4], d=order, which='Chebyshev', where=key)
                if grid.is_tensor:
                    assert vals.data.dim() == 3
                    if order == 0:
                        assert vals.data.shape == torch.Size([16, len(value(grid)), 1])
                    else:
                        assert vals.data.shape == torch.Size([16, len(value(grid)), 2])

                # Test the function values
                grid_rescaled = rescale_grid(grid, new_domain=[[-1, 1], [-1, 1]],
                                             as_tensor=grid.is_tensor, requires_grad=grid.requires_grad)

                if order > 0:
                    for k in range(len(vals.data)):
                        for p in range(len(value(grid))):
                            tf_val = vals.data[k][p]
                            test_val = [
                                dtest_function(value(grid_rescaled).data[p][0], int(vals.coords[k][0].detach().numpy()),
                                               d=order, which='Chebyshev') *
                                dtest_function(value(grid_rescaled).data[p][1], int(vals.coords[k][1].detach().numpy()),
                                               d=0, which='Chebyshev'),
                                dtest_function(value(grid_rescaled).data[p][0], int(vals.coords[k][0].detach().numpy()),
                                               d=0, which='Chebyshev') *
                                dtest_function(value(grid_rescaled).data[p][1], int(vals.coords[k][1].detach().numpy()),
                                               d=order, which='Chebyshev')]
                            test_val = torch.reshape(torch.stack(test_val), (2, ))
                            assert (tf_val == test_val).all()





