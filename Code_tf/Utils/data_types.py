from typing import Sequence, Union
import tensorflow as tf
import numpy as np

"""Custom data types used in the VPINNS code"""


class Grid:
    """Grid type. A grid consists of pairs of x- and y values, which may have
    different dimensions.
    """
    def __init__(self, *, x: Sequence[float], y: Sequence[float] = None,
                 as_tensor: bool = False, dtype: tf.DType = tf.dtypes.float64):
        """Constructs a grid from coordinate arrays
        x: the x coordinates
        y: the y coordinates
        as_tensor: whether to return the grid objects as tf.Tensors
        dtpye: the data type to use
        Raises:
            ValueError: if x is empty.
        """
        if x is None or not x:
            raise ValueError('Cannot create an empty grid!')
        if not as_tensor:
            self._x = x
            self._y = y if y is not None else []
        else:
            self._x = [tf.constant([pt], dtype=dtype) for pt in x]
            self._y = [] if y is None else [tf.constant([pt], dtype=dtype) for pt in y]

        if y is None:
            if not as_tensor:
                self._data = [[p] for p in x]
                self._boundary = [x[0], x[-1]]
            else:
                self._data = [tf.constant([p], dtype=dtype) for p in x]
                self._boundary = [tf.constant(x[0], dtype=dtype), tf.constant(x[-1], dtype=dtype)]
        else:
            data = []
            for y_val in y:
                for x_val in x:
                    if not as_tensor:
                        data.append([x_val, y_val])
                    else:
                        data.append(tf.constant([[x_val, y_val]], dtype=dtype))
            self._data = data
            boundary = []
            for p in x:
                if not as_tensor:
                    boundary.append([p, y[0]])
                    boundary.append([p, y[-1]])
                else:
                    boundary.append(tf.constant([[p, y[0]]], dtype=dtype))
                    boundary.append(tf.constant([[p, y[-1]]], dtype=dtype))
            for p in y:
                if not as_tensor:
                    boundary.append([x[0], p])
                    boundary.append([x[-1], p])
                else:
                    boundary.append(tf.constant([[x[0], p]], dtype=dtype))
                    boundary.append(tf.constant([[x[-1], p]], dtype=dtype))
            self._boundary = boundary
        self._size = len(self._data)

    # .. Magic methods ........................................................

    def __getitem__(self, item):
        return self._data[item]

    def __iter__(self):
        for i in range(self._size):
            yield self._data[i]

    def __str__(self) -> str:
        output = f"Grid of size ({len(self._x)} x {len(self._y)})"
        if self._x:
            output += f"; x interval: [{np.around(np.min(self._x), 4)}, {np.around(np.max(self._x), 4)}]"
        if self._y:
            output += f"; y interval: [{np.around(np.min(self._y), 4)}, {np.around(np.max(self._y), 4)}]"
        return output

    # .. Properties ...........................................................

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def data(self):
        return self._data

    @property
    def boundary(self):
        return self._boundary

    @property
    def dim(self):
        return (self._x != [])+(self._y != [])

    @property
    def size(self):
        return self._size


class DataSet:
    """DataSet class. A DataSet consists of grid points and matching function values. These can be either lists
     or TensorFlow tf.Tensor objects. The DataSet can return information on its size, the dimension of the
     underlying coordinates, and the coordinate data for a given coordinate axis.
    """

    def __init__(self, *, x: Sequence, f: Sequence[float],
                 as_tensor: bool = False, data_type: tf.DType = tf.dtypes.float64):
        """Initializes a DataSet object from a list of coordinates and corresponding data values.

        Args:
            x: the coordinates
            f: the data values
            as_tensor: whether to return the DataSet as containing tf.Tensor objects
            data_type: the data type of the tf.Tensor objects

        Raises:
            ValueError: when the coordinates and function values do not have equal dimensions
            ValueError: when trying to initialize an empty DataSet
        """
        if len(x) != len(f):
            raise ValueError("x and f must be of same dimension!")
        if not x:
            raise ValueError("Cannot generate empty Dataset!")

        if isinstance(x[0], Sequence):
            coords = []
            for val in x:
                if as_tensor:
                    pt = tf.constant([[val[i] for i in range(len(val))]], dtype=data_type)
                else:
                    pt = [val[i] for i in range(len(val))]
                coords.append(pt)
            self._coords = coords
            self._dim = len(x[0])
        else:
            if as_tensor:
                self._coords = [tf.constant([[val]], dtype=data_type) for val in x]
            else:
                self._coords = x
            self._dim = 1
        if as_tensor:
            self._data = [tf.constant([[val]], dtype=data_type) for val in f]
        else:
            self._data = f
        self._size = len(f)

    # .. Magic methods ........................................................

    def __getitem__(self, item):
        return self._coords[item], self._data[item]

    def __iter__(self):
        for i in range(self._size):
            yield self._coords[i], self._data[i]

    def __str__(self) -> str:
        return (f"Dataset of {len(self._coords)} function values on a "
                f"{len(self._dim)}-dimensional grid")

    # .. Properties ...........................................................

    # Get a particular coordinate dimension
    def axis(self, ax: int):
        if self._dim == 1:
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
    def dim(self):
        return self._dim


class DataGrid:
    """DataGrid type. A DataGrid consists of a grid and matching function values."""

    def __init__(self, *, x: Grid, f: Sequence[float],
                 as_tensor: bool = False, dtype: tf.DType = tf.dtypes.float64):
        """Initializes a DataGrid object from a grid and corresponding data values.

        Args:
            x: the grid
            f: the data values
        Raises:
            ValueError: when the coordinates and function values do not have equal dimensions
            ValueError: when trying to initialize an empty DataSet
        """
        if x.size != len(f):
            raise ValueError("x and f must be of same dimension!")
        if not x:
            raise ValueError("Cannot generate empty DataGrid!")

        self._grid = x
        self._x = x.x
        self._y = x.y
        self._dim = x.dim
        self._size = len(f)
        self._data = f if not as_tensor else [tf.constant([val], dtype=dtype) for val in f]

    # .. Magic methods ........................................................

    def __iter__(self):
        for i in range(self._size):
            yield self._grid.data[i], self._data[i]

    def __getitem__(self, item):
        return self._grid.data[item], self._data[item]

    def __str__(self) -> str:
        return (f"DataGrid on {self._grid.size} grid points on a {self._dim}-dimensional "
                f"grid")

    # .. Properties ...........................................................

    # Get the underlying grid
    @property
    def grid(self):
        return self._grid

    # Get the function values
    @property
    def data(self):
        return self._data

    # Get the x axis of the grid
    @property
    def x(self):
        return self._x

    # Get the y axis of the grid
    @property
    def y(self):
        return self._y

    # Get the grid dimension
    @property
    def dim(self):
        return self._dim

    # Get the size of the dataset
    @property
    def size(self):
        return self._size
