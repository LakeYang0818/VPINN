import numpy as np
import torch
from typing import Sequence, Union
import warnings


class Grid:
    """
    Grid type. A grid consists of pairs of x- and y values, which may have
    different dimensions. The Grid class contains information on its boundary,
    its interior points, as well its dimension and size. Can be 1D or 2D.
    """

    def __init__(self, x: Sequence, y: Sequence = None,
                 *, as_tensor: bool = True, dtype=torch.float, requires_grad: bool = False):

        """Constructs a Grid object.

        :param x: the x coordinates
        :param y: the y coordinates. Can be None.
        :param as_tensor: whether to return the Grid as tensor points
        :param dtype: the data type to use
        :param requires_grad: whether the grid points require gradients, if torch.Tensors are being used

        Raises:
            ValueError: if x is empty
            ValueError: if y is not None but empty
            ValueError: if x and y are of different types
            Warning: if a tensor argument is passed, but 'as_tensor' is set to false. In this case, a
            tensor will still be returned
            Warning: if 'as_tensor' is False and 'requires_grad' is True. In this case, 'requires_grad' will
            be set to false.
        """

        # Ascertain valid and compatible arguments
        if x is None:
            raise ValueError('Cannot create an empty grid!')

        if len(x) == 0:
            raise ValueError(f"Invalid arg 'x' for grid: cannot generate grid from empty {type(x)}.")
        else:
            self._n_x = len(x)

        if isinstance(x, torch.Tensor):
            if x.dim() > 2:
                raise ValueError(f"Invalid arg 'x' for grid: cannot generate grid from {x.dim()}-dimensional tensor.")
            elif not as_tensor:
                warnings.warn(f"You have passed a {type(x)} argument for 'x' but set 'as_tensor = False'. "
                              f"In this case, 'as_tensor' will be set to true.")
                self._as_tensor = True

        if requires_grad and not as_tensor:
            warnings.warn("You have set 'as_tensor' to False but 'requires_grad' to 'True'. 'requires_grad' is only"
                          " effective for torch.Tensor types, and will be ignored. Alternatively, set "
                          "'as_tensor' to 'True' or pass torch.Tensor types as coordinate arguments.")
            requires_grad = False

        if y is not None:
            if type(x) != type(y):
                raise ValueError(f"Arguments 'x' and 'y' must be of same type, but are of types "
                                 f"{type(x)} and {type(y)}")
            if isinstance(y, torch.Tensor):
                if y.size() == torch.Size([0]):
                    raise ValueError(f"Invalid arg 'y' for grid: cannot generate grid from empty tensor.")
                elif y.dim() > 2:
                    raise ValueError(
                        f"Invalid arg 'y' for grid: cannot generate grid from {y.dim()}-dimensional tensor.")
                elif not as_tensor:
                    warnings.warn(f"You have passed a {type(y)} argument for 'y' but set 'as_tensor = False'. "
                                  f"In this case, 'as_tensor' will be set to true.")
                    as_tensor = True
                else:
                    self._n_y = y.size()[0]
            else:
                if len(y) == 0:
                    raise ValueError(f"Invalid arg 'y' for grid: cannot generate grid from empty {type(y)}.")
                else:
                    self._n_y = len(x)
            self._dim = 2
        else:
            self._dim = 1

        # Create coordinate axes
        if not as_tensor:
            self._x = np.array(x).astype(dtype)
            self._n_x = len(self._x)
            self._y = np.array(y).astype(dtype) if y is not None else np.array([])
            self._n_y = len(self._y)
            self._dim = 1 + (self._n_y > 1)
        else:
            if isinstance(x, torch.Tensor):
                self._x = torch.reshape(torch.tensor(x.detach().clone().numpy(), dtype=dtype), (self._n_x, 1))
            else:
                self._x = torch.reshape(torch.tensor(x, dtype=dtype), (self._n_x, 1))

            if y is not None:
                if isinstance(y, torch.Tensor):
                    self._y = torch.reshape(torch.tensor(x.detach().clone().numpy(), dtype=dtype),
                                            (self._n_y, 1))
                else:
                    self._y = torch.reshape(torch.tensor(x, dtype=dtype), (self._n_y, 1))

        # Create datasets
        if y is None:
            if not as_tensor:
                self._data = np.resize(np.array([val for val in x]), (self._n_x, 1))
                self._boundary = np.resize(np.array([x[0], x[-1]]), (2, 1))
                self._interior = np.resize(np.array([val] for val in x[1:-1]), (self._n_x - 2, 1))
            else:
                self._data = self._x
                self._boundary = torch.reshape(torch.stack([self._x[0], self._x[-1]]), (2, 1))
                self._interior = torch.reshape(self._x[1:-1], (self._n_x - 2, 1))
            self._size = self._n_x
        else:
            data, boundary, interior = [], [], []
            for j in range(self._n_y):
                for i in range(self._n_x):
                    if not as_tensor:
                        point = [x[i], y[j]]
                    else:
                        point = torch.tensor([x[i], y[j]], dtype=dtype)
                        point = torch.reshape(point, (1, 2))
                    data.append(point)
                    if i == 0 or i == (self._n_x - 1) or j == 0 or j == (self._n_y - 1):
                        boundary.append(point)
                    else:
                        interior.append(point)
            if not as_tensor:
                self._data = np.resize(np.array(data), (self._n_x * self._n_y, 2))
                self._boundary = np.resize(np.array(boundary), (2 * (self._n_x + self._n_y - 2), 2))
                self._interior = np.resize(np.array(interior), ((self._n_x - 2) * (self._n_y - 2), 2))
            else:
                self._data = torch.reshape(torch.stack(data), (self._n_x * self._n_y, 2))
                self._boundary = torch.reshape(torch.stack(boundary), (2 * (self._n_x + self._n_y - 2), 2))
                self._interior = torch.reshape(torch.stack(interior), ((self._n_x - 2) * (self._n_y - 2), 2))

            self._size = self._n_x * self._n_y

        # Set requires_gradient flag, if required
        if as_tensor and requires_grad:
            self._x.requires_grad = requires_grad
            if y is not None:
                self._y.requires_grad = requires_grad
            self._data.requires_grad = requires_grad
            self._boundary.requires_grad = requires_grad
            self._interior.requires_grad = requires_grad

        self._as_tensor = as_tensor

    # .. Magic methods .................................................................................................

    def __getitem__(self, item):
        return self._data[item]

    def __iter__(self):
        for i in range(self._size):
            yield self._data[i]

    def __str__(self) -> str:
        output = f"Grid of size ({self._n_x} x {self._n_x})"
        if self._x:
            output += f"; x interval: [{np.around(min(self._x)[0], 4)}, {np.around(max(self._x)[0], 4)}]"
        if self._y:
            output += f"; y interval: [{np.around(min(self._y)[0], 4)}, {np.around(max(self._y)[0], 4)}]"
        return output

    # .. Properties ....................................................................................................

    @property
    def boundary(self):
        return self._boundary

    @property
    def data(self):
        return self._data

    @property
    def dim(self):
        return self._dim

    @property
    def interior(self):
        return self._interior

    @property
    def is_tensor(self):
        return self._as_tensor

    @property
    def size(self):
        return self._size

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y


# ... Grid constructor ................................................................................................

def construct_grid(dim: int,
                   boundary: Sequence,
                   grid_size: Union[int, Sequence[int]],
                   *,
                   as_tensor: bool = True,
                   dtype=torch.float,
                   requires_grad: bool = False) -> Grid:
    """Constructs a grid of the given dimension.

    :param dim: the dimension of the grid
    :param boundary: the boundaries of the grid
    :param grid_size: the number of grid points in each dimension
    :param as_tensor: whether to return the grid points as tensors
    :param dtype: the datatype of the grid points
    :param requires_grad: whether the grid points require differentiability
    :return: the grid
    """

    def _grid_1d(lower: float, upper: float, n_points: int) -> Sequence:

        """Constructs a one-dimensional grid."""

        step_size = (1.0 * upper - lower) / (1.0 * n_points - 1)

        return [lower + _ * step_size for _ in range(n_points)]

    if dim == 1:

        return Grid(x=_grid_1d(boundary[0], boundary[1], grid_size),
                    as_tensor=as_tensor, dtype=dtype, requires_grad=requires_grad)

    elif dim == 2:
        x: Sequence = _grid_1d(boundary[0][0], boundary[0][1], grid_size[0])
        y: Sequence = _grid_1d(boundary[1][0], boundary[1][1], grid_size[1])

        return Grid(x=x, y=y, as_tensor=as_tensor, dtype=dtype, requires_grad=requires_grad)
