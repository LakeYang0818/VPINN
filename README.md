# VPINNs: Variational Physics-Informed Neural Nets
### Daniel Boutros, Thomas Gaskin, Oscar de Wit

---

This tool uses a variational physics-informed neural network to learn weak solutions for non-linear
PDEs. Our code covers one-, two-, and three-dimensional square domains without mesh refinement.

The model uses the [utopya]() package for simulation control and configuration. It can be controlled from
a configuration file, with data handling and plotting automatically taken care of. Concurrent runs (*parameter sweeps*)
are easily configurable from the configuration files, and are automatically parallelised. The neural core itself
is implemented using [Pytorch](https://pytorch.org/tutorials/) and is unit tested.

> **_Note_**: This README gives a brief introduction to installation and running a model, as well as a basic
> overview of the Utopia syntax. You can find a complete guide on running models with Utopia/utopya
> [here](https://docs.utopia-project.org/html/getting_started/tutorial.html#tutorial).

## How to install
#### 1. Clone this repository
Clone this repository using a link obtained from 'Code' button (for non-developers, use HTTPS):

```console
git clone <GIT-CLONE-URL>
```
If you are a developer, [get an SSH key registered](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) and clone with SSH.

#### 2. Install requirements
We recommend creating a new virtual environment in a location of your choice and installing all requirements into the
venv. The following command will install the [utopya package](https://gitlab.com/utopia-project/utopya) and the utopya CLI
from [PyPI](https://pypi.org/project/utopya/), as well as all other requirements:

```console
pip install -r requirements.txt
```
This assumes your current directory is the project folder. You should now be able to invoke the utopya CLI:
```console
utopya --help
```
> **_Note_**  On Apple Silicon devices running macOS 12.3+, follow [these](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/)
> instructions to install pytorch with GPU training enabled. Your training device will then be set to 'mps'.
> On all devices the GPU, where available, will always be the preferred training device.

#### 3. Register the project and the model with utopya
In the project directory (i.e. this one), register the entire project using the following command:
```console
utopya projects register .
```
You should get a positive response from the utopya CLI and your project should appear in the project list when calling:
```console
utopya projects ls
```
> **_Note_** Any changes to the project info file need to be communicated to utopya by calling the registration command anew.
> You will then have to additionally pass the `````--exists-action overwrite````` flag, because a project of that name already exists.
> See ```utopya projects register --help``` for more information.

Finally, register the model via
```console
utopya models register from-manifest model/VPINN_info.yml
```
Done! 🎉


## How to run a model
Now you have set up the model, run it by invoking
```console
utopya run VPINN
```

### TODO ....
The  model will generate a synthetic dataset of test function and grid data, train the neural net for 1000 epochs
(default value), and write the output into a new directory, located in your home directory
`~/utopya_output` by default.

The default configuration settings are provided in the `HarrisWilson_cfg.yml` file in the
`models/HarrisWilson` folder. You can modify the settings here, but we recommend changing the configuration
settings by instead creating a `run.yml` file somewhere and using it to run the model. You can do so by
calling
```console
utopya run HarrisWilson path/to/run_cfg.yml
```
In this file, you only need to specify those entries from the `<modelname>_cfg.yml` file you wish to change,
and not reproduce the entire configuration set. The advantage of this approach is that you can
create multiple configs for different scenarios, and leave the working base configuration untouched.
An example could look like this:

```yaml
parameter_space:
  seed: 4
  num_epochs: 3000
  write_start: 1
  write_every: 1
  HarrisWilson:
    Data:
      synthetic_data:
        alpha: 1.5
        beta: 4.2
        sigma: 0.1
```
This is generating a synthetic dataset using all the settings from the `HarrisWilson_cfg.yml` file *except* for those
You can run the model using this file by calling
```console
utopya run HarrisWilson path/to/cfg.yml
```

> **_Note_**: The models all come with plenty of example configuration files in the `cfgs` folders. These are
> *configuration sets*, complete sets of run configurations and evaluation routines designed to produce specific
> plots. These also demonstrate how to load datasets to run the models.

## Parameter sweeps
> **_Note_**: Take a look at the [full tutorial entry](https://docs.utopia-project.org/html/getting_started/tutorial.html#parameter-sweeps)
> for a full guide on running parameter sweeps.

Parameter sweeps (multiple runs using different configuration settings) are easy: all you need to do is add a
`!sweep` tag to all parameters you wish to sweep over. Parameter sweeps are automatically run in parallel.
For example, to sweep over the `seed` (to generate some statistics, say), just do

```yaml
parameter_space:
  seed: !sweep
    default: 0
    range: [10]
```
Then call your model via

```console
utopya run <model_name> --run-mode sweep
```
The model will then run ten times, each time using a different seed value. You can also add the following entry to
the configuration file at the root-level:
```yaml
perform_sweep: True
```
You can then run a sweep without the ``--run-mode`` flag in the CLI.
Passing a `default` argument to the sweep parameter(s) is required: this way, the model can still perform a single run
when a sweep is not configured. Again, there are plenty of examples in the `cfgs` folders.


## Running a model using configuration sets
Configuration sets are a useful way of gathering a combination of run settings and plot configurations
in a single place, so as to automatically generate data and plots that form a set.
The `HarrisWilson` model contains a large number of *configuration sets* comprising run configs and *evaluation* configs,
that is, plot configurations. These sets will reproduce the plots from the publication.
You can run them by executing

```console
utopya run HarrisWilson --cfg-set <name_of_cfg_set>
```

> **_Note_** Some of the configuration sets perform *sweeps*, that is, runs over several parameter configurations.
> These may take a while to run.

Running the configuration set will produce plots. If you wish to re-evaluate a run (perhaps plotting different figures),
you do not need to re-run the model, since the data has already been generated. Simply call

```console
utopya eval HarrisWilson --cfg-set <name_of_cfg_set>
```

This will re-evaluate the *last model you ran*. You can re-evaluate any dataset, of course, by
providing the path to that dataset, like so:

```console
utopya eval HarrisWilson path/to/output/folder --cfg-set <name_of_cfg_set>
```
## How to adjust the neural net configurations
You can vary the size of the neural net and the activation functions
right from the config. The size of the input layer is inferred from
the data passed to it, and the size of the output layer is
determined by the number of parameters you wish to learn — all the hidden layers
can be determined by the user. The net is configured from the ``NeuralNet`` key of the
config:

```yaml
NeuralNet:
  num_layers: 6
  nodes_per_layer: 20
  activation_funcs:
    first: sine
    2: cosine
    3: tanh
    last: abs
  bias: True
  init_bias: [0, 4]
  learning_rate: 0.002
```
``num_layers`` and ``nodes_per_layer`` give the structure of the hidden layers (hidden layers
with different numbers of nodes is not yet supported). The ``activation_funcs`` dictionary
allows specifying the activation function on each layer: just add the number of the layer together
with the name of a common function, such as ``relu``, ``linear``, ``tanh``, ``sigmoid``, etc.
``bias`` controls use of the bias, and the ``init_bias`` sets the initialisation interval for the
bias.

## Training settings
You can modify the training settings, such as the batch size or the training device, from the
`Training` entry of the config:

```yaml
Training:
  batch_size: 1
  to_learn: [ param1, param2, param3 ]
  true_parameters:
    param4: 0.5
  device: cpu
  num_threads: ~
```
The `to_learn` entry lists the parameters you wish to learn. If you are not learning the complete
parameter set, you must supply the parameter value to use during training for that parameter under
`true_parameters`.

The `device` entry sets the training device. The default here is the `cpu`; you can set it to any
supported pytorch training device. Make sure your platform is configured to support the selected device.
On Apple Silicon, set the device to `mps` to enable GPU training, provided you have followed the corresponding
installation instructions (see above).

`utopya` automatically parallelises multiple runs; the number of CPU cores available to do this
can be specified under `worker_managers/num_workers` on the root-level configuration (i.e. on the same level as
`parameter_space`). The `Training/num_threads` entry controls the number of threads *per model run* to be used during training.
If you thus set `num_workers` to 4 and `num_threads` to 3, you will in total be able to use 12 threads.

## Loading data
See the model-specific README files to see how to load different types of data. Data is stored in the `data/`
folder.

## 🚧 Tests (WIP)
To run tests, invoke
```bash
pytest tests
```
from the main folder.




### Required packages

The following packages are required to run the code. We recommend installing these
into a virtal environment to avoid interfering with system-wide package installations.

| Package    | Version  | Comments                            |
|------------|----------|-------------------------------------|
| Python     | \>= 3.9  |                                     |
| Pytorch    | \>= 1.10 | Machine learning package.           |
| PyYAML     | \>= 6.0  | Handles configuration               |
| matplotlib |          | Handles plotting                    |
| latex      |          | Required for plotting               |
| pytest     |          | Optional; used for running tests    |
| pandas     |          | Optional; used for writing out data |

> **_Note:_**  On Apple Silicon, using the GPU to train is currently [WIP](https://github.com/pytorch/pytorch/issues/47702).
>
### Running the model
To execute, run the `main.py` file. Results are stored in the `Results` folder — make sure not to delete it.
A jupyter notebook is also provided.
#### Modifying model parameters
All settings should be modified from the `config.yml` file. The following is a list of the relevant parameters:

| Parameter                         | Explanation                                                                                                                                                                                                |
|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `dimension`                       | The dimension of the underlying space. Can be 1 or 2. For the Burgers equation, this must <br/>be 2.                                                                                                       |
| `boundary`                        | The boundary of the grid. If the `dim` is 1, this should a 1d array; if `dim` is 2, this should be an array of arrays, ie. ```[0, 1]``` for a 1-d grid from 0 to 1, ```[[0, 1], [0, 1]]``` for a 2-d grid. |
| `grid_size`                       | The number of grid points in each dimension.                                                                                                                                                               |
| `PDE/type`                        | The differential equation to use. Can be any of `Poisson`, `Helmholtz`, `Burger`.                                                                                                                          |
| `Helmholtz/k`                     | The `k`-parameter of the Helmholtz equation.                                                                                                                                                               |
| `Burger/nu`                       | The viscosity term in the Burger's equation.                                                                                                                                                               |
| `PorousMedium/m`                  | The `m`-parameter of the porous medium equation.                                                                                                                                                           |
| `variational_form`                | Which variational form to use for the variational loss. Can be any of `1`, `2`, `3`.                                                                                                                       |
| `Test functions/N_test_functions` | The number of test functions against which to integrate in each dimension                                                                                                                                  |
| `Test functions/Type`             | The type of test function to use: can be `Legendre`, `Chebyshev`, or `Sine`                                                                                                                                |
| `Test functions/weighting`        | Whether to weight the test functions by a factor of 2^(-k) in the loss function                                                                                                                            |
| `architecture/layers`             | Number of hidden layers in the neural net.                                                                                                                                                                 |
| `architecture/nodes_per_layer`    | Size of the hidden layers.                                                                                                                                                                                 |
| `N_iterations`                    | Number of epochs.                                                                                                                                                                                          |
| `learning_rate`                   | Learning rate of the SGD.                                                                                                                                                                                  |
| `boundary_loss_weight`            | Relative weight of the boundary loss. A value of `0` turns off the boundary loss.                                                                                                                          |
| `variational_loss_weight`         | Relative weight of the variational loss. A value of `0` turns off the variational loss.                                                                                                                    |
| `plot_resolution`                 | Plot resolution in each coordinate dimension                                                                                                                                                               |
| `plot_animation`                  | For 2D models: plot an animation of the developement over time                                                                                                                                             |
| `plot_info_box`                   | Write out the configuration parameters into an info box in the plots.                                                                                                                                      |
| `write_loss_data`                 | Write out the loss data as a csv file for later plotting. Requires pandas.                                                                                                                                 |
| `rcParams`                        | rcParams for the plots                                                                                                                                                                                     |

#### Modifying the external forcing
To modify the external forcing, go to the `function_definitions.py` file and modify the function called `f`. Several
examples from the report are already provided – simply set the `example` string to the entry of the dictionary
provided to run the example you wish. Test functions can be modified in `Utils/test_functions`. One-dimensional
test functions must take two inputs: an x-value, and an `int n` which serves as
a label for the test function.

#### Adding new equation types
To let the model solve new equation types, add a new file to the `Utils/Variational_forms.py` folder, and include it in the
module by adding it to `Variational_forms/__init__.py`. Return your new
variational loss in the `variational_loss` function in the `VPINN` class (`VPINN.py`).

> **_Note:_** It is essential that all your datatypes have the same _dimensions_ when calculating
> the losses. If your loss functions stay constant even for a high number of iterations,
> this may be indicative of incompatible data dimensions, e.g. between training data and model
> predictions. pytorch will not throw an error, but will simply not be able to correctly perform
> the backwards pass through the loss calculation.

> **_Hint:_** We recommend making use of the _testing_ capabilities when implementing new
> features (see below). Use the testing tools provided to first write comprehensive tests for whatever it is you wish to implement,
> thereby making sure your feature behaves as expected. This will significantly reduce the time you
> spend debugging the code.
>
### Custom data types

We have implemented several custom data types to facilitate handling
multidimensional data, as well as data on grids. These types are implemented in the `Utils/Datatypes`
folder. They are all subscriptable and iterable.

#### The `Grid` class:
```python
class Grid(*, x: Sequence = None, y: Sequence = None, as_tensor: bool = True, dtype=torch.float,
          requires_grad: bool = False, requires_normals: bool = False)
```
This implements a grid from a list of coordinates. A grid can be either one- or two-dimensional,
depending on how many arguments are passed. If ```as_tensor``` is set,
the grid points will be `torch.Tensors`. If `requires_grad` is set, the
grid points will be trackable. Grid attributes can be easily accessed via several useful
member functions:

- ```grid.x``` and ```grid.y``` return the coordinate axes, while ```grid.data```
returns the grid as an array of points.
- ```grid.boundary``` returns the grid boundary. The grid boundary is oriented in the mathematically positive sense, ie. counterclockwise.
- You can access individual sections of the boundary for a 2D grid via
```grid.lower_boundary```, ```grid.upper_boundary```, ```grid.right_boundary```, ```grid.left_boundary```.
- ```grid.dim``` returns the
dimension of the grid, and ```grid.size``` returns the number of points it contains.
- ``grid.volume`` returns the grid volume, ```grid.boundary_volume``` returns the
length of the boundary.
- ```grid.normals``` returns the sign of each boundary element. This is only returned
if the grid was created with the ```requires_normals``` argument set.


#### The `DataSet` class:
```python
class DataSet(*, coords: Sequence, data: Sequence,
              as_tensor: bool = False, data_type: torch.dype = torch.float)
```
The `DataSet` class is similar in syntax to `pandas.DataFrame`, but allowing for compatibility with
tensorflow types. A `DataSet` contains points and data values --- the former can be accessed via
`DataSet.coords`, the latter via `DataSet.data`. If `as_tensor` is set to `True`, the elements will be
`torch.Tensors`, allowing for them to be passed to a neural net. The `DataSet` can return only the coordinates of
a particular axis via `DataSet.axis()`, as well as the shorthands `DataSet.x` and `DataSet.y` for convenience.
`DataSet.dim` and `DataSet.size` returns the dimensions of the coordinates and the size of the data set respectively.

### Running tests

The code is unit-tested to ensure correct functionality. It is recommended that you test
any changes or additions you make to the package. To run tests, navigate to the folder containing the `main.py` file, enter your
virtual environment, and execute `python -m pytest -v Tests/`. This will run *all* tests in the `Tests` folder.
To run individual tests, simply specify the file you wish to run.
