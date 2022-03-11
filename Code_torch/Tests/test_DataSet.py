import numpy as np
import torch

from Utils.Datatypes.Grid import Grid
from Utils.Datatypes.DataSet import DataSet
from Utils import test_function as tfunc, dtest_function


"""Unit tests for the Grid Datatype"""

# To do: test higher dimensional data

# Test the grid initialisation
def test_DataSet_initialization():

    # Test 1d non-tensor type DataSet initialization
    coords1d = [
        np.linspace(-0.8, 0.8, 11),
        [2, 3, 4, 5],
        Grid(x=np.linspace(1, 2, 4), dtype=np.float64, as_tensor=False).x
    ]

    for coord in coords1d:
        func_vals = dtest_function(coord, 1, d=1)
        test_data_set = DataSet(coords=coord, data=func_vals, dtype=np.float64, as_tensor=False)
        assert (test_data_set.size == len(coord))
        assert (test_data_set.dim == 1)
        assert (test_data_set.coords == np.reshape(coord, (len(coord), 1))).all()
        assert (test_data_set.data == np.reshape(func_vals, (len(func_vals), 1))).all()

    coords1d_tensor = [
        torch.from_numpy(np.linspace(-0.8, 0.8, 11)),
        torch.stack([torch.tensor(2), torch.tensor(3)]),
        torch.stack([torch.tensor([2]), torch.tensor([4])]),
        Grid(x=np.linspace(1, 2, 4)).x,
        Grid(x=np.linspace(1, 2, 4), requires_grad = True).x,

    ]

    # Test 1d tensor type DataSet initialization
    for coord in coords1d_tensor:
        func_vals = dtest_function(coord, 1, d=1)
        test_data_set = DataSet(coords=coord, data=func_vals)
        assert (test_data_set.size == len(coord))
        assert (test_data_set.dim == 1)
        assert (test_data_set.coords == coord).all()
        assert (test_data_set.data == torch.reshape(func_vals, (len(func_vals), 1))).all()

    # Test 2d non-tensor type DataSet initialization
    coords2d = [
        [[2, 0], [3, 1], [4, 1], [5, 4]],
        Grid(x=np.linspace(1, 2, 4), y=np.linspace(0.3, 0.6, 3), dtype=np.float64, as_tensor=False).data
    ]

    for coord in coords2d:

        # vector-valued function data
        func_vals = dtest_function(coord, 1, d=1)
        test_data_set = DataSet(coords=coord, data=func_vals, dtype=np.float64, as_tensor=False)
        assert (test_data_set.size == len(coord))
        assert (test_data_set.dim == 2)
        assert (test_data_set.coords == coord).all()
        assert (test_data_set.data == np.reshape(func_vals, (len(func_vals), 2))).all()

        # scalar-valued function data
        func_vals = tfunc(coord, 1)
        test_data_set = DataSet(coords=coord, data=func_vals, dtype=np.float64, as_tensor=False)
        assert (test_data_set.data == np.reshape(func_vals, (len(func_vals), 1))).all()

    coords2d_tensor = [
        torch.stack([torch.tensor([2, 3]), torch.tensor([3, 4])]),
        Grid(x=np.linspace(1, 2, 4), y=np.linspace(0.3, 0.6, 3)).data,
        Grid(x=np.linspace(1, 2, 4), y=np.linspace(0.3, 0.6, 3), requires_grad=True).data
    ]

    # Test 1d tensor type DataSet initialization
    for coord in coords2d_tensor:

        # vector-valed function data
        func_vals = dtest_function(coord, 1, d=1)
        test_data_set = DataSet(coords=coord, data=func_vals)
        assert (test_data_set.size == len(coord))
        assert (test_data_set.dim == 2)
        assert (test_data_set.coords == coord).all()
        assert (test_data_set.data == torch.reshape(func_vals, (len(func_vals), 2))).all()

        # scalar-valued function data
        func_vals = tfunc(coord, 1)
        test_data_set = DataSet(coords=coord, data=func_vals)
        assert (test_data_set.data == np.reshape(func_vals, (len(func_vals), 1))).all()