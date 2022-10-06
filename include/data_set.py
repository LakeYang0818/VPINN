import numbers
import numpy as np
import torch
from typing import Any
import warnings


class DataSet:

    """DataSet class. A DataSet consists of points and matching function values. The DataSet can return information on
    its size, the dimension of the underlying coordinates, and the coordinate data for a given coordinate axis.
    """


    def __init__(self,
                 *,
                 coords: Any,
                 data: Any,
                 as_tensor: bool = True,
                 dtype=torch.float,
                 requires_grad: bool = False):

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
        if (len(coords) != len(data)):
            raise ValueError('Coords and DataSet must have equal length!')
        if isinstance(coords, torch.Tensor):
            if not as_tensor:
                warnings.warn(f"You have passed {type(coords)} arguments but set 'as_tensor = False'. "
                              f"In this case, 'as_tensor' will be set to true.")
                as_tensor = True

        if requires_grad and not as_tensor:
            warnings.warn("You have set 'as_tensor' to False but 'requires_grad' to 'True'. 'requires_grad' is only"
                          " effective for torch.Tensor types, and will be ignored. Alternatively, set "
                          "'as_tensor' to 'True' or pass torch.Tensor types as coordinate arguments.")
            requires_grad = False

        self._size = len(coords)

        # Set the coordinate attributes
        if not as_tensor:
            self._coords = np.array(coords)
        else:
            if isinstance(coords, torch.Tensor):
                self._coords = coords.detach().clone()
                self._coords.requires_grad = requires_grad
            else:
                self._coords = torch.stack(
                    [torch.tensor(val, dtype=dtype, requires_grad=requires_grad) for val in coords])

        # Set the data values
        if not as_tensor:
            self._data = np.array(data)
        else:
            if isinstance(data, torch.Tensor):
                  self._data = data.detach().clone()
                  self._data.requires_grad = requires_grad
            else:
                if isinstance(data[0], torch.Tensor):
                    self._data = torch.stack([torch.tensor(val.clone().detach().numpy(), dtype=dtype,
                                                           requires_grad=requires_grad) for val in data])
                else:
                    self._data = torch.stack([torch.tensor(val, dtype=dtype, requires_grad=requires_grad) for val in data])


    # .. Magic methods .................................................................................................

    def __getitem__(self, item):
        return self._coords[item], self._data[item]

    def __iter__(self):
        for i in range(self._size):
            yield self._coords[i], self._data[i]

    def __str__(self) -> str:
        return (f'Dataset of {len(self._coords)} {len(self._data[0])}-dimensional function values on '
                f'{int(np.prod(np.shape(self._coords[-1])))}-dimensional coordinates.')

    # .. Properties ....................................................................................................

    # Get a particular coordinate dimension
    def axis(self, ax: int):
        if ax >= np.shape(self._coords)[0]:
            raise ValueError(f'Axis {ax} not retrieavable: '
                             f'DataSet coordinates have dimension {np.shape(self._coords[0])}!')
        if np.shape(self._coords[0]) <= (1, ):
            return self._coords
        else:
            return np.swapaxes(self._coords, 0, 1)[ax]

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
            return int(np.prod(np.shape(self._coords[0])))
        elif which == 'data':
            return int(np.prod(np.shape(self.data[0])))
        else:
            raise ValueError(f"Unrecognized argument {which}!")
