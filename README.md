# VPINN: Variational physics-informed neural net 
### Daniel Boutros, Thomas Gaskin, Oscar de Wit

A machine learning module for solving weak non-linear PDEs, developed for the CMI
core course project at the University of Cambridge. It is originally based on [code](https://github.com/ehsankharazmi/hp-VPINNs) 
by Ehsan Kharazmi et al., who initially designed a machine learning tool to solve 
non-linear PDEs with weak solutions. We have rewritten the code and extended it to also cover 2-dimensional domains, 
and have added several additional differential equations that can be solved. The model can be controlled from a config file. 
We use [TensorFlow 2](https://www.tensorflow.org/guide), allowing among other things 
the use of [`eager execution`](https://www.tensorflow.org/guide/function) (WIP).


### Required packages

The following packages are required to run the code. We recommend installing these
into a virtal environment to avoid interfering with system-wide package installations.

| Package        | Version | Comments                   |
|----------------|---------|----------------------------|
| Python         | \>= 3.9 |                            |
| Tensorflow     | \>= 2.0 | Machine learning package.  |
| PyYAML         | \>= 6.0 | Handles configuration      |
| pyDOE          | 
| matplotlib

> **_Note:_**  On Apple Silicon, install tensorflow via `pip install tensorflow-macos`. 
> If you wish to train on your GPU, try following [these](https://developer.apple.com/metal/tensorflow-plugin/) 
> instructions (although we have not managed to do so.)
> 
### How to run
To execute, run the `main.py` file.
#### Modifying model parameters
All settings should be modified from the `config.yml` file. The following is a list of the relevant parameters:

| Parameter                      | Explanation                                                                                                                                                                                                   |
|--------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `dimension`                    | The dimension of the underlying space. Can be 1 or 2. For the Burgers equation, this must <br/>be 2.                                                                                                          |
| `boundary`                     | The boundary of the grid. If the `dim` is 1, this should a 1d array; if `dim` is 2, this should be an array of arrays, ie. ```[0, 1]``` for a 1-d grid from 0 to 1, ```[[0, 1], [0, 1]]``` for a 2-d grid.    |
| `grid_size`                    | The number of grid points in each dimension. Should be a scalar for `dim: 1`, an array for `dim: 2`.                                                                                                          |
| `PDE/type`                     | The differential equation to use. Can be any of `Poisson`, `Helmholtz`, `Burger`.                                                                                                                             |
| `Helmholtz/k`                  | The `k`-parameter of the Helmholtz equation.                                                                                                                                                                  |
| `Burger/nu`                    | The viscosity term in the Burger's equation.                                                                                                                                                                  |
| `variational_form`             | Which variational form to use for the variational loss. Can be any of `1`, `2`, `3`.                                                                                                                          |
| `N_test_functions`             | The number of test functions against which to integrate. Should be a scalar.                                                                                                                                  |
| `N_quad`                       | The number of quadrature points in each dimension. Should be a scalar for `dim: 1`, an array for `dim: 2`.                                                                                                    |
| `N_f`                          | Plot resolution (WIP)                                                                                                                                                                                         |
| `architecture/layers`          | Number of hidden layers in the neural net.                                                                                                                                                                    |
| `architecture/nodes_per_layer` | Size of the hidden layers.                                                                                                                                                                                    |
| `N_iterations`                 | Number of epochs.                                                                                                                                                                                             |
| `learning_rate`                | Learning rate of the SGD.                                                                                                                                                                                     |
| `loss_weight`                  | Relative weight of the variational loss to strong residual. A value of `0` turns off the variational term.                                                                                                    |
| `batch_size`                   | Number of training batches per epoch.                                                                                                                                                                         |

#### Modifying the external forcing
To modify the external forcing, go to the `Utils/functions.py` file and modify the function called `f`. Test functions can also 
be modified in that file. Test functions must take two inputs: an x-value, and an `int n` which serves as 
a label for the test function.

### About the code

**Custom data types:** We have implemented several custom data types to facilitate handling
multidimensional data, as well as data on grids. These types are implemented in the `Utils/data_types.py`
file. They are all subscriptable and iterable.

#### The `Grid` class:
```python
class Grid(*, x: Sequence = None, y: Sequence = None)
```
This implements a grid from a list of coordinates. A grid can be either one- or two-dimensional, 
depending on how many arguments are passed. Grid attributes can be easily accessed via several useful
member functions: ```grid.x``` and ```grid.y``` return the coordinate axes, while ```grid.data```
returns the grid as an array of points. ```grid.boundary``` returns the grid boundary. ```grid.dim``` returns the 
dimension of the grid, and ```grid.size``` returns the number of points it contains.

#### The `DataSet` class:
```python
class DataSet(*, x: Sequence, f: Sequence,
              as_tensor: bool = False, data_type: tf.DType = tf.dtypes.float64)
```
The `DataSet` class is similar in syntax to `pandas.DataFrame`, but allowing for compatibility with 
tensorflow types. A `DataSet` contains points and data values --- the former can be accessed via 
`DataSet.coords`, the latter via `DataSet.data`. If `as_tensor` is set to `True`, the elements will be 
`tf.Tensors`, allowing for them to be passed to a neural net. The `DataSet` can return only the coordinates of 
a particular axis via `DataSet.axis()`, as well as the shorthands `DataSet.x` and `DataSet.y` for convenience.
`DataSet.dim` and `DataSet.size` returns the dimensions of the coordinates and the size of the data set respectively.

#### The `DataGrid` class:
```python
class DataGrid(*, x: Grid, f: Sequence)
```
Combines function values on an underlying `Grid`. Access the grid via `DataGrid.grid` and the data values via 
`DataGrid.data`. As before, `DataGrid.x`, `DataGrid.y`, `DataGrid.dim` and `DataGrid.size` are all possible.
