# VPINN: Variational physics-informed neural net 
### Daniel Boutros, Thomas Gaskin, Oscar de Wit

A machine learning tool for solving weak non-linear PDEs, developed for the CMI
core course project at the University of Cambridge. It is based on [work](https://doi.org/10.1016/j.cma.2020.113547) 
by Ehsan Kharazmi et al. Our code covers 1- and 2-dimensional domains, 
and includes several additional differential equations that can be solved. The model can be controlled from a config file. 
The code is implemented using [Pytorch](https://pytorch.org/tutorials/).


### Required packages

The following packages are required to run the code. We recommend installing these
into a virtal environment to avoid interfering with system-wide package installations.

| Package    | Version  | Comments                   |
|------------|----------|----------------------------|
| Python     | \>= 3.9  |                            |
| Pytorch    | \>= 1.10 | Machine learning package.  |
| PyYAML     | \>= 6.0  | Handles configuration      |
| matplotlib |          | Handles plotting           |
| latex      |          | Recommended for plotting   |

> **_Note:_**  On Apple Silicon, using GPU to train is currently [WIP](https://github.com/pytorch/pytorch/issues/47702).
> 
### How to run
To execute, run the `main.py` file.
#### Modifying model parameters
All settings should be modified from the `config.yml` file. The following is a list of the relevant parameters:

| Parameter                      | Explanation                                                                                                                                                                                                |
|--------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `dimension`                    | The dimension of the underlying space. Can be 1 or 2. For the Burgers equation, this must <br/>be 2.                                                                                                       |
| `boundary`                     | The boundary of the grid. If the `dim` is 1, this should a 1d array; if `dim` is 2, this should be an array of arrays, ie. ```[0, 1]``` for a 1-d grid from 0 to 1, ```[[0, 1], [0, 1]]``` for a 2-d grid. |
| `grid_size`                    | The number of grid points in each dimension.                                                                                                                                                               |
| `PDE/type`                     | The differential equation to use. Can be any of `Poisson`, `Helmholtz`, `Burger`.                                                                                                                          |
| `Helmholtz/k`                  | The `k`-parameter of the Helmholtz equation.                                                                                                                                                               |
| `Burger/nu`                    | The viscosity term in the Burger's equation.                                                                                                                                                               |
| `variational_form`             | Which variational form to use for the variational loss. Can be any of `1`, `2`, `3`.                                                                                                                       |
| `N_test_functions`             | The number of test functions against which to integrate in each dimension                                                                                                                                  |
| `architecture/layers`          | Number of hidden layers in the neural net.                                                                                                                                                                 |
| `architecture/nodes_per_layer` | Size of the hidden layers.                                                                                                                                                                                 |
| `N_iterations`                 | Number of epochs.                                                                                                                                                                                          |
| `learning_rate`                | Learning rate of the SGD.                                                                                                                                                                                  |
| `boundary_loss_weight`         | Relative weight of the boundary loss. A value of `0` turns off the boundary loss.                                                                                                                          |
| `variational_loss_weight`      | Relative weight of the variational loss. A value of `0` turns off the variational loss.                                                                                                                    |
| `plot_res`                       | Plot resolution in each coordinate dimension                                                                                                                                                               |
| `rcParams`                       | rcParams for the plots                                                                                                                                                                                     |

#### Modifying the external forcing
To modify the external forcing, go to the `function_definitions.py` file and modify the function called `f`. Test functions can 
be modified in `Utils/test_functions`. One-dimensional test functions must take two inputs: an x-value, and an `int n` which serves as 
a label for the test function.

### About the code

**Custom data types:** We have implemented several custom data types to facilitate handling
multidimensional data, as well as data on grids. These types are implemented in the `Utils/Types`
folder. They are all subscriptable and iterable.

#### The `Grid` class:
```python
class Grid(*, x: Sequence = None, y: Sequence = None)
```
This implements a grid from a list of coordinates. A grid can be either one- or two-dimensional, 
depending on how many arguments are passed. Grid attributes can be easily accessed via several useful
member functions: ```grid.x``` and ```grid.y``` return the coordinate axes, while ```grid.data```
returns the grid as an array of points. ```grid.boundary``` returns the grid boundary. ```grid.dim``` returns the 
dimension of the grid, and ```grid.size``` returns the number of points it contains. ``grid.volume`` returns the 
grid volume.

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
