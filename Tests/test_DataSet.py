import numpy as np
import torch

from Utils.Datatypes.Grid import Grid
from Utils.Datatypes.DataSet import DataSet
from Utils import dtest_function


"""Unit tests for the Grid Datatype"""


# Test the grid initialisation
def test_DataSet_initialization_1d():

    # One-dimensional DataSet initialization
    coords = [
        np.linspace(-0.8, 0.8, 11),
        [2, 3, 4, 5],
        Grid(x=np.linspace(1, 2, 4), dtype=np.float64, as_tensor=False).x,
        torch.from_numpy(np.linspace(-0.8, 0.8, 11)),
        torch.stack([torch.tensor(2), torch.tensor(3)]),
        torch.stack([torch.tensor([2]), torch.tensor([4])]),
        Grid(x=np.linspace(1, 2, 4)).x,
        Grid(x=np.linspace(1, 2, 4), requires_grad=True).x
    ]

    for coord in coords:
        func_vals = dtest_function(coord, 1, d=1)
        as_tensor = isinstance(coord, torch.Tensor) or (isinstance(coord, Grid) and Grid.is_tensor)
        test_dataset = DataSet(coords=coord, data=func_vals, dtype=type(coord), as_tensor=as_tensor)

        assert (test_dataset.size == len(coord))
        assert (test_dataset.dim == 1)
        assert (test_dataset.coords == coord).all()
        assert (test_dataset.data == func_vals).all()

# Two-dimensional type DataSet initialization
def test_DataSet_initialization_2d():

    coords = [
        [[2, 0], [3, 1], [4, 1], [5, 4]],
        Grid(x=np.linspace(1, 2, 4), y=np.linspace(0.3, 0.6, 3), dtype=np.float64, as_tensor=False).data,
        torch.stack([torch.tensor([2, 3]), torch.tensor([3, 4])]),
        Grid(x=np.linspace(1, 2, 4), y=np.linspace(0.3, 0.6, 3)).data,
        Grid(x=np.linspace(1, 2, 4), y=np.linspace(0.3, 0.6, 3), as_tensor=True, requires_grad=True).data
    ]

    for coord in coords:

        # vector-valued function data
        func_vals = dtest_function(coord, 1, d=1)
        as_tensor = isinstance(coord, torch.Tensor) or (isinstance(coord, Grid) and Grid.is_tensor)
        test_dataset = DataSet(coords=coord, data=func_vals, dtype=type(coord), as_tensor=as_tensor)
        assert (test_dataset.size == len(coord))
        assert (test_dataset.dim == 2)
        assert (test_dataset.coords == coord).all()
        assert (test_dataset.data == func_vals).all()

        # scalar-valued function data
        func_vals = dtest_function(coord, 1, d=0)
        test_dataset = DataSet(coords=coord, data=func_vals, dtype=np.float64, as_tensor=as_tensor)
        assert (test_dataset.dim == 2)
        assert (test_dataset.coords == coord).all()
        assert (test_dataset.data == func_vals).all()

# Three-dimensional type DataSet initialization
def test_DataSet_initialization_3d():
    
    coords = np.arange(1, 11, 1)
    x = Grid(x=np.linspace(-1, 1, 6))
    data = [dtest_function(x.x, i, d=1) for i in range(10)]
    test_dataset = DataSet(coords=coords, data=data)

    assert (test_dataset.dim == 1)
    assert (test_dataset.size == 10)
