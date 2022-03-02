import numpy as np
import torch
from typing import Any
import warnings


class DataSet:

    """DataSet class. A DataSet consists of points and matching function values. The DataSet can return information on
    its size, the dimension of the underlying coordinates, and the coordinate data for a given coordinate axis.
    """

    def __init__(self, *, coords: Any, data: Any,
                 as_tensor: bool = True, dtype=torch.float, requires_grad: bool = False):

        """Constructs a DataSet object

        :param coords: the coordinate points
        :param data: the function values
        :param as_tensor: whether to return the dataset using torch.Tensor objects
        :param dtype: the data type to use
        :param requires_grad: whether the grid points require gradients, if torch.Tensors are being used

        Raises:
            ValueError: if coords are empty
            ValueError: if data values are empty
            ValueError: if coords and data are of different types
            ValueError: if coords and data are of different length
            Warning: if a tensor argument is passed, but 'as_tensor' is set to false. In this case, a
            tensor will still be returned
            Warning: if 'as_tensor' is False and 'requires_grad' is True. In this case, 'requires_grad' will
            be set to false.
        """

        # Ascertain valid and compatible arguments
        if coords is None:
            raise ValueError('Cannot create DataSet with NoneType coordinates!')
        if data is None:
            raise ValueError('Cannot create DataSet with NoneType data values!')
        if type(coords) != type(data):
            raise ValueError(f"Arguments 'coords' and 'data' must be of same type, but are of types "
                             f"{type(coords)} and {type(data)}")

        if len(coords) != len(data):
            raise ValueError('DataSet coordinate and data dimensions must be equal!')

        if isinstance(coords, torch.Tensor):
            if coords.size() == torch.Size([0]):
                raise ValueError('Cannot create DataSet from empty coordinates!')
            if coords.dim() > 2:
                raise ValueError(f"Invalid arg 'coords': cannot generate DataSet from "
                                 f"{coords.dim()}-dimensional coordinates.")
            if coords.dim() != data.dim():
                raise ValueError(f"Coordinates and data values must have same shape!")
            if data.size() == torch.Size([0]):
                raise ValueError('Cannot create DataSet from empty data values!')
            if not as_tensor:
                warnings.warn(f"You have passed {type(coords)} arguments but set 'as_tensor = False'. "
                              f"In this case, 'as_tensor' will be set to true.")
                as_tensor = True

        if requires_grad and not as_tensor:
            warnings.warn("You have set 'as_tensor' to False but 'requires_grad' to 'True'. 'requires_grad' is only"
                          " effective for torch.Tensor types, and will be ignored. Alternatively, set "
                          "'as_tensor' to 'True' or pass torch.Tensor types as coordinate arguments.")
            requires_grad = False

        self._dim_coords = len(coords[0])
        self._dim_data = len(data[0])
        self._size = len(coords)

        # Set the coordinate attributes
        if not as_tensor:
            self._coords = np.resize(np.array(coords), (len(coords), self._dim_coords)).astype(dtype)
        else:
            if isinstance(coords, torch.Tensor):
                self._coords = coords

            else:
                coords = [torch.tensor(val, dtype=dtype, requires_grad=requires_grad) for val in coords]
                self._coords = torch.reshape(torch.stack(coords), (len(coords), self._dim_coords))

        # Set the data values
        if not as_tensor:
            self._data = np.resize(np.array(data), (len(coords), self._dim_data)).astype(dtype)
        else:

            if isinstance(data, torch.Tensor):
                  self._data = torch.reshape(data,(len(coords), self._dim_data))
            else:
                # Note: for some odd reason, reshaping here leads to a breaking change. Don't know why
                if isinstance(data[0], torch.Tensor):
                    self._data = torch.stack([torch.tensor(val.clone().detach().numpy(), dtype=dtype, requires_grad=requires_grad) for val in data])
                else:
                    self._data = torch.stack([torch.tensor(val, dtype=dtype, requires_grad=requires_grad) for val in data])

        # Set requires_grad attributes for torch.Tensors
        if as_tensor and requires_grad:
            self._coords.requires_grad = requires_grad
            self._data.requires_grad = requires_grad

    # .. Magic methods .................................................................................................

    def __getitem__(self, item):
        return self._coords[item], self._data[item]

    def __iter__(self):
        for i in range(self._size):
            yield self._coords[i], self._data[i]

    def __str__(self) -> str:
        return (f"Dataset of {len(self._coords)} {self._dim_data}-dimensional function values on a "
                f"{self._dim_coords}-dimensional grid")

    # .. Properties ....................................................................................................

    # Get a particular coordinate dimension
    def axis(self, ax: int):
        if self._dim_coords == 1:
            if ax == 1:
                return self._coords
            else:
                return []
        else:
            return [[p[ax] for p in self._coords]]

    # Get the coordinates
    @property
    def coords(self):
        return self._coords

    # Get the data values
    @property
    def data(self):
        return self._data

    # Shorthand: return the x-coordinates
    @property
    def x(self):
        return self.axis(1)

    # Shorthand: return the y-coordinates
    @property
    def y(self):
        return self.axis(2)

    # Return the number of data values
    @property
    def size(self):
        return self._size

    # Return the dimension of the underlying coordinates
    @property
    def dim(self, which: str = 'coords'):
        if which == 'coords':
            return self._dim_coords
        elif which == 'data':
            return self._dim_data
        else:
            raise ValueError(f"Unrecognized argument {which}!")
