import numpy as np
import xarray


def construct_grid(space_dict: dict, *, lower: int = None, upper: int = None, dtype = None) -> xarray.DataArray:

    """ Constructs a grid from a configuration dictionary, and returns it as a xarray.DataArray.

    :param space_dict: the dictionary of space configurations
    :param lower: (optional) the lower grid value to use if no extent is given
    :param upper: (optional) the upper grid value to use if no extent is given. Defaults to the extent of the grid.
    :param dtype: (optional) the grid value datatype to use
    :return: the grid (an xarray.DataArray)

    raises:
        ValueError: for grid dimensions of size greater than 3 (currently not implemented)
    """

    coords, data_vars = {}, {}

    # Create the coordinate dictionary
    for key, entry in space_dict.items():
        l, u = entry.pop('extent', (lower, upper))
        u = entry['size'] if u is None else u
        coords[key] = (key, np.linspace(l, u, entry['size'], dtype=dtype))

    # Create the meshgrid. The order of the indices needs to be adapted to the output shape of np.meshgrid.
    indices = list(coords.keys()) + ['idx']
    if len(indices) == 2:
        data = coords[indices[0]][1]

    elif len(indices) == 3:

        x, y = np.meshgrid(coords[indices[0]][1], coords[indices[1]][1])
        indices[0], indices[1] = indices[1], indices[0]
        data = np.stack([x, y], axis=-1)

    elif len(indices) == 4:
        x, y, z = np.meshgrid(coords[indices[0]][1], coords[indices[1]][1], coords[indices[2]][1])
        indices[0], indices[1] = indices[1], indices[0]
        data = np.stack([x, y, z], axis=-1)

    else:
        raise ValueError(f'Currently not implemented for grid dimensions > 4; got dimension {len(indices)-1}!')

    # Update the DataArray coordinates
    coords.update(dict(idx=('idx', np.arange(1, len(indices), 1))))

    # Add grid dimension as an attribute
    res = xarray.DataArray(coords=coords, data=data, dims=indices)
    res.attrs['grid_dimension'] = len(indices) - 1

    # Calculate grid density for integration
    n_points, volume = np.prod(list(res.sizes.values()))/res.sizes['idx'], 1
    for key, range in res.coords.items():
        if key == 'idx':
            continue
        volume *= (range.isel({key: -1}) - range.isel({key: 0}))

    res.attrs['grid_density'] = volume / n_points

    return res


from typing import Sequence, Union, Any
import torch
import warnings

class Grid:

    def __init__(self,
                 x: Sequence,
                 y: Sequence = None,
                 *,
                 as_tensor: bool = True,
                 dtype=torch.float,
                 requires_grad: bool = False,
                 requires_normals: bool = False):

        """Constructs a Grid object. A grid consists of pairs of x- and y values, which may have
        different dimensions. The Grid class contains information on its boundary,
        its interior points, as well its dimension and size. Can be 1D or 2D.

        :param x: the x coordinates
        :param y: the y coordinates. Can be None.
        :param as_tensor: whether to return the Grid as points of torch.Tensors
        :param dtype: the data type to use
        :param requires_grad: whether the grid points require gradients, if torch.Tensors are being used
        :paran requires_normals: whether to calculate the grid boundary normal vectors

        :raises ValueError: if x and y are of different types
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
                self._n_y = y.size()[0]
            else:
                if len(y) == 0:
                    raise ValueError(f"Invalid arg 'y' for grid: cannot generate grid from empty {type(y)}.")
                else:
                    self._n_y = len(y)
            self._dim = 2
        else:
            self._dim = 1

        # Create coordinate axes
        if not as_tensor:
            self._x = np.resize(np.array(x).astype(dtype), (self._n_x, 1))
            self._n_x = len(self._x)
            self._y = np.resize(np.array(y).astype(dtype), (self._n_y, 1)) if y is not None else np.array([])
            self._n_y = len(self._y)
            self._dim = 1 + (self._n_y > 1)
        else:
            if isinstance(x, torch.Tensor):
                self._x = torch.reshape(torch.tensor(x.detach().clone().numpy(), dtype=dtype), (self._n_x, 1))
            else:
                self._x = torch.reshape(torch.tensor(x, dtype=dtype), (self._n_x, 1))

            if y is not None:
                if isinstance(y, torch.Tensor):
                    self._y = torch.reshape(torch.tensor(y.detach().clone().numpy(), dtype=dtype),
                                            (self._n_y, 1))
                else:
                    self._y = torch.reshape(torch.tensor(y, dtype=dtype), (self._n_y, 1))

        # Create datasets
        if y is None:
            if not as_tensor:
                self._data = np.resize(self._x, (self._n_x, 1))
                self._boundary = np.resize(np.array([self._x[0], self._x[-1]]), (2, 1))
                self._interior = np.resize(self._x[-1:1], (self._n_x - 2, 1))
            else:
                self._data = self._x
                self._boundary = torch.reshape(torch.stack([self._x[0], self._x[-1]]), (2, 1))
                self._interior = torch.reshape(self._x[1:-1], (self._n_x - 2, 1))
            self._size = self._n_x
        else:

            # Collect data and interior points
            data, interior = [], []
            for j in range(self._n_y):
                for i in range(self._n_x):
                    if not as_tensor:
                        point = [self._x[i], self._y[j]]
                    else:
                        point = torch.reshape(torch.stack([self._x[i], self._y[j]]), (1, 2))
                    data.append(point)
                    if i == 0 or i == (self._n_x - 1) or j == 0 or j == (self._n_y - 1):
                        continue
                    else:
                        interior.append(point)

            #  Collect boundary points. The boundary is the contour of the domain, oriented counter-clockwise
            lower, right, upper, left = [], [], [], []
            for i in range(self._n_x):
                if not as_tensor:
                    lower.append([self._x[i], self._y[0]])
                    upper.append([self._x[len(self._x) - i - 1], self._y[-1]])
                else:
                    lower.append(torch.reshape(torch.stack([self._x[i], self._y[0]]), (1, 2)))
                    upper.append(torch.reshape(torch.stack([self._x[len(self._x) - i - 1], self._y[-1]]), (1, 2)))
            for j in range(self._n_y):
                if not as_tensor:
                    right.append([self._x[-1], self._y[j]])
                    left.append([self._x[0], self._y[len(self._y) - j - 1]])
                else:
                    right.append(torch.reshape(torch.stack([self._x[-1], self._y[j]]), (1, 2)))
                    left.append(torch.reshape(torch.stack([self._x[0], self._y[len(self._y) - j - 1]]), (1, 2)))
            boundary = lower[:-1] + right[:-1] + upper[:-1] + left[:-1]

            if not as_tensor:
                self._data = np.resize(np.array(data), (self._n_x * self._n_y, 2))
                self._boundary = np.resize(np.array(boundary), (2 * (self._n_x + self._n_y - 2), 2))
                self._lower = np.resize(np.array(lower), (self._n_x, 2))
                self._right = np.resize(np.array(right), (self._n_y, 2))
                self._upper = np.resize(np.array(upper), (self._n_x, 2))
                self._left = np.resize(np.array(left), (self._n_y, 2))
                self._interior = np.resize(np.array(interior),
                                           ((self._n_x - 2) * (self._n_y - 2), 2)) if interior != [] else []
            else:
                self._data = torch.reshape(torch.stack(data), (self._n_x * self._n_y, 2))
                self._boundary = torch.reshape(torch.stack(boundary), (2 * (self._n_x + self._n_y - 2), 2))
                self._lower = torch.reshape(torch.stack(lower), (self._n_x, 2))
                self._right = torch.reshape(torch.stack(right), (self._n_y, 2))
                self._upper = torch.reshape(torch.stack(upper), (self._n_x, 2))
                self._left = torch.reshape(torch.stack(left), (self._n_y, 2))
                self._interior = torch.reshape(torch.stack(interior),
                                               ((self._n_x - 2) * (self._n_y - 2), 2)) if interior != [] else []

            self._size = self._n_x * self._n_y

        # Get the grid boundary normals
        if requires_normals:
            if self._dim == 1:
                if as_tensor:
                    self._normals = torch.stack([torch.tensor([-1], dtype=dtype), torch.tensor([1], dtype=dtype)])
                else:
                    self._normals = np.array([[-1], [1]])
            else:
                if as_tensor:
                    self._normals = torch.stack(
                        [torch.tensor([0, 1])] * (len(self._x) - 1) +
                        [torch.tensor([-1, 0])] * (len(self._y) - 1) +
                        [torch.tensor([0, -1])] * (len(self._x) - 1) +
                        [torch.tensor([1, 0])] * (len(self._y) - 1)
                    )
                else:
                    self._normals = np.array(
                        [[0, 1]] * (len(self._x) - 1) +
                        [[-1, 0]] * (len(self._y) - 1) +
                        [[0, -1]] * (len(self._x) - 1) +
                        [[1, 0]] * (len(self._y) - 1)
                    )
        else:
            self._normals = None

        # Set requires_gradient flag, if required
        if as_tensor and requires_grad:
            self._x.requires_grad = requires_grad
            if y is not None:
                self._y.requires_grad = requires_grad
            self._data.requires_grad = requires_grad
            self._boundary.requires_grad = requires_grad
            self._interior.requires_grad = requires_grad

        self._as_tensor = as_tensor
        self._dtype = dtype
        self._req_grad = requires_grad

    # .. Magic methods .................................................................................................

    def __getitem__(self, item):
        return self._data[item]

    def __iter__(self):
        for i in range(self._size):
            yield self._data[i]

    def __str__(self) -> str:
        output = f"Grid of size {self._n_x}"
        if self._dim == 2:
            output += f" x {self._n_y}"
        output += f"; x interval: [{np.around(min(self._x), 4)}, {np.around(max(self._x), 4)}]"
        if self._dim == 2:
            output += f"; y interval: [{np.around(min(self._y), 4)}, {np.around(max(self._y), 4)}]"
        return output

    # .. Properties ....................................................................................................

    @property
    def boundary(self):
        return self._boundary

    @property
    def lower_boundary(self):
        if self.dim == 1:
            raise ValueError("1-dimensional grid has no attribute 'lower_boundary'!")
        else:
            return self._lower

    @property
    def upper_boundary(self):
        if self.dim == 1:
            raise ValueError("1-dimensional grid has no attribute 'upper_boundary'!")
        else:
            return self._upper

    @property
    def right_boundary(self):
        if self.dim == 1:
            raise ValueError("1-dimensional grid has no attribute 'right_boundary'!")
        else:
            return self._right

    @property
    def left_boundary(self):
        if self.dim == 1:
            raise ValueError("1-dimensional grid has no attribute 'left_boundary'!")
        else:
            return self._left


    @property
    def boundary_volume(self):
        if self.dim == 1:
            return self._x[-1] - self._x[0]
        elif self.dim == 2:
            return 2 * (self._x[-1] - self._x[0]) + 2 * (self._y[-1] - self._y[0])

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
    def normals(self):
        return self._normals

    @property
    def size(self):
        return self._size

    @property
    def volume(self):
        if self.dim == 1:
            return self._x[-1] - self._x[0]
        elif self.dim == 2:
            return (self._x[-1] - self._x[0]) * (self._y[-1] - self._y[0])

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def dtype(self):
        return self._dtype

    @property
    def requires_grad(self):
        return self._req_grad


# ... Grid constructor ................................................................................................

# def construct_grid(*, dim: int,
#                    boundary: Sequence,
#                    grid_size: Union[int, Sequence[int]],
#                    as_tensor: bool = True,
#                    dtype=torch.float,
#                    requires_grad: bool = False,
#                    requires_normals: bool = False) -> Grid:
#     """Constructs a grid of the given dimension.
#
#     :param dim: the dimension of the grid
#     :param boundary: the boundaries of the grid
#     :param grid_size: the number of grid points in each dimension
#     :param as_tensor: whether to return the grid points as tensors
#     :param dtype: the datatype of the grid points
#     :param requires_grad: whether the grid points require differentiability
#     :param requires_normals: whether the grid boundary normal values need to be calculated
#     :return: the grid
#     """
#
#     def _grid_1d(lower: float, upper: float, n_points: int) -> Sequence:
#
#         """Constructs a one-dimensional grid."""
#
#         step_size = (1.0 * upper - lower) / (1.0 * n_points - 1)
#
#         return [lower + _ * step_size for _ in range(n_points)]
#
#     if dim == 1:
#
#         return Grid(x=_grid_1d(boundary[0], boundary[1], grid_size),
#                     as_tensor=as_tensor, dtype=dtype, requires_grad=requires_grad, requires_normals=requires_normals)
#
#     elif dim == 2:
#         x: Sequence = _grid_1d(boundary[0][0], boundary[0][1], grid_size[0])
#         y: Sequence = _grid_1d(boundary[1][0], boundary[1][1], grid_size[1])
#
#         return Grid(x=x, y=y, as_tensor=as_tensor, dtype=dtype, requires_grad=requires_grad,
#                     requires_normals=requires_normals)
#
#
# def rescale_grid(grid: Grid, *, new_domain) -> Grid:
#     """ Rescales a grid to a new domain.
#
#     :param grid: the grid to rescale
#     :param new_domain: the domain to which to scale the grid
#     :return: the rescaled grid
#
#     :raises ValueError: if a grid is being rescaled to a new domain whose dimension does not match the
#         grid dimension.
#     """
#
#
#     if grid.dim == 1:
#         if np.shape(new_domain) != (2,):
#             raise ValueError(f"Cannot rescale 1d grid to {len(np.shape(new_domain))}-dimensional domain!")
#
#         return Grid(x=(new_domain[1] - new_domain[0]) * (grid.x - grid.x[0]) / (grid.x[-1] - grid.x[0]) + new_domain[0],
#                     as_tensor=grid.is_tensor, dtype=grid.dtype, requires_grad=grid.requires_grad, requires_normals=(grid.normals is not None))
#
#     elif grid.dim == 2:
#         if np.shape(new_domain) != (2, 2):
#             raise ValueError(f"Cannot rescale 2d grid to {len(np.shape(new_domain))}-dimensional domain!")
#
#         return Grid(
#             x=(new_domain[0][1] - new_domain[0][0]) * (grid.x - grid.x[0]) / (grid.x[-1] - grid.x[0]) + new_domain[0][
#                 0],
#             y=(new_domain[1][1] - new_domain[1][0]) * (grid.y - grid.y[0]) / (grid.y[-1] - grid.y[0]) + new_domain[1][
#                 0],
#             as_tensor=grid.is_tensor, dtype=grid.dtype, requires_grad=grid.requires_grad, requires_normals=(grid.normals is not None))
