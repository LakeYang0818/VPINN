import numpy as np
import pytest
import torch

from Utils.Datatypes.Grid import construct_grid, Grid, rescale_grid

"""Unit tests for the Grid Datatype"""

# To do: test case where interior is empty

# Test the grid initialisation
def test_Grid_initialization():

    # Non-tensor grid type
    x = np.linspace(0., 1., 3)
    y = np.linspace(-1., 1., 3)

    g1 = Grid(x=x, y=y, as_tensor=False, dtype=np.float32)
    assert (g1.size == len(x) * len(y))
    assert (g1.volume == (x[-1] - x[0]) * (y[-1] - y[0]))
    assert (g1.dim == 2)
    assert (g1.interior == [[0.5, 0.0]]).all()
    assert (g1.x == np.resize(x, (len(x), 1))).all()
    assert (g1.y == np.resize(y, (len(y), 1))).all()
    assert g1.volume == 2
    assert g1.boundary_volume == 6
    for p in g1.data:
        if p not in g1.interior:
            assert (p in g1.boundary)

    assert (g1.boundary == [[0, -1], [0.5, -1], [1, -1], [1, 0], [1, 1], [0.5, 1], [0, 1], [0., 0]]).all()

    g2 = Grid(x=g1.x, y=g1.y, as_tensor=False, dtype=np.float32)
    assert (g2.x == g1.x).all()
    assert (g2.y == g1.y).all()
    assert (g2.data == g1.data).all()
    assert (not g2.is_tensor)
    assert g2.volume == 2
    assert g2.boundary_volume == 6

    # Tensor grid type
    g3 = Grid(x=g2.x, as_tensor=True)
    assert (g3.dim == 1)
    assert (g3.x.size() == torch.Size([3, 1]))
    assert g3.is_tensor

    g4 = Grid(x=g1.x, y=g1.y, dtype=torch.float, requires_grad=True)

    assert (g4.dim == g1.dim)
    assert (g4.size == g1.size)
    assert (g4.interior.detach().numpy() == g1.interior).all()
    assert (g4.data.detach().numpy() == g1.data).all()
    assert (g4.boundary.detach().numpy() == g1.boundary).all()
    assert (g4.dtype == torch.float)
    assert g4.is_tensor

    g4 = Grid(x=g4.x, y=g4.y)

    g5 = Grid(x=g1.x, y=g1.y, as_tensor=True, dtype=torch.int)
    g6 = Grid(x=g1.x, y=g1.y, as_tensor=False, dtype=int)

    assert (g5.dim == g6.dim)
    assert (g5.size == g6.size)
    assert (g5.interior.numpy() == g6.interior).all()
    assert (g5.data.numpy() == g6.data).all()
    assert (g5.boundary.numpy() == g6.boundary).all()

    grids = [g1, g2, g3, g4, g5, g6]

    # Test printing and iteration
    for grid in grids:
        assert (str(grid))
        for _, p in enumerate(grid):
            assert (p == p).all()


# Test the grid construction
def test_Grid_construction():
    grid1 = construct_grid(dim=1, boundary=[-1, 1], grid_size=20, as_tensor=False, dtype=np.float64)
    assert (grid1.size == 20)
    assert (grid1.dim == 1)
    assert grid1.volume == 2
    assert grid1.boundary_volume == 2

    grid2 = construct_grid(dim=2, boundary=[[-1, 1], [0, 10]], grid_size=[5, 7])
    assert (grid2.size == 35)
    assert (grid2.dim == 2)
    assert grid2.volume == 20
    assert grid2.boundary_volume == 24


# Test grid rescaling to a new domain
def test_Grid_rescaling():
    grid1 = construct_grid(dim=2, boundary=[[-4, 4], [0, 10]], grid_size=[9, 19], as_tensor=False, dtype=np.float64)
    grid2 = rescale_grid(grid1, new_domain=[[-1, 1], [-1, 1]])

    assert (grid2.size == grid1.size)
    assert (grid2.dim == grid1.dim)

    assert (grid2.boundary[0] == [-1, -1]).all()

    with pytest.raises(ValueError):
        _ = rescale_grid(grid2, new_domain=[-1, 1])

    grid3 = rescale_grid(grid2, new_domain=[[-1, 1], [-1, 1]])
    assert (grid3.size == grid2.size)

    for i in range(len(grid3.data)):
        assert (grid3.data[i] == grid2.data[i]).all()

    grid4 = construct_grid(dim=2, boundary=[[-4, 4], [0, 10]], grid_size=[10, 13], requires_grad=True)
    grid5 = rescale_grid(grid4, new_domain=[[-1, 1], [-1, 1]])

    assert (grid4.boundary.detach().numpy()[9] == [4, 0]).all()
    assert (grid5.boundary.detach().numpy()[9] == [1, -1]).all()
    assert grid5.is_tensor
    assert grid5.requires_grad
    assert grid5.data[0].requires_grad
    assert (grid4.size == grid5.size)

    grid6 = rescale_grid(grid4, new_domain = [[-1, 1], [0, 1]], requires_grad=False)
    assert grid6.is_tensor
    assert not grid6.requires_grad
    assert not grid6.data[0].requires_grad
    assert (grid4.size == grid6.size)