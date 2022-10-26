# VPINNs: Variational Physics-Informed Neural Nets
### Daniel Boutros, Thomas Gaskin, Oscar de Wit

---

This tool uses a variational physics-informed neural network to learn weak solutions for non-linear
PDEs. Our code allows for one-, two-, and three-dimensional square domains without mesh refinement.

The model uses the [utopya]() package for simulation control and configuration. It can be controlled from
a configuration file, with data handling and plotting automatically taken care of. Concurrent runs (*parameter sweeps*)
are easily configurable from the configuration files, and are automatically parallelised. The neural core itself
is implemented using [Pytorch](https://pytorch.org/tutorials/) and is unit tested.

> **_Note_**: This README gives a brief introduction to installation and running a model, as well as a basic
> overview of the Utopia syntax. You can find a complete guide on running models with Utopia/utopya
> [here](https://docs.utopia-project.org/html/getting_started/tutorial.html#tutorial).

## How to install
#### 1. Clone this repository
Clone this repository using a link obtained from 'Code' button:

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
> For ease of use and to ensure cross-platform compatibility, the CPU is set as the default training device.

#### 3. Register the project and the model with utopya
In the project directory (i.e. this one), register the entire project using the following command:
```console
utopya projects register .
```
You should get a positive response from the utopya CLI and your project should appear in the project list when calling:
```console
utopya projects ls
```
> **_Note_** Should you make any changes to the `VPINN_info.yml` file, these will need to be communicated to utopya by
> calling the registration command anew.
> You will then have to additionally pass the `````--exists-action overwrite````` flag, because a project of that name already exists.
> See ```utopya projects register --help``` for more information.

Finally, register the model via
```console
utopya models register from-manifest model/VPINN_info.yml
```
Done! 🎉

#### 4. Download data using git LFS (optional)
The repository contains some pre-processed grid and test function data, which speeds up running models considerably,
since you do not need to generate the training data. These files are stored using [git LFS](https://git-lfs.github.com)
(large file storage). To download them, first install git lfs via
```console
git lfs install
```
This assumes you have the git command line extension installed. Then, from within the repo, do
```console
git lfs pull
```
This will pull all the datasets.

## How to run a model
Now you have set up the model, run it by invoking the basic run command
```console
utopya run VPINN
```
This will run the model using the default configuration settings.
It will generate a synthetic dataset of test function and grid data, train the neural net for 100 epochs
(default value), and write the output into a new directory, located in your home directory at
`~/utopya_output` by default.

The default configuration settings are provided in the `VPINN_cfg.yml` file in the
`model` folder. You can modify the settings here, but we recommend changing the configuration
settings by instead creating a `run.yml` file somewhere and using that to run the model. You can do so by
calling
```console
utopya run VPINN path/to/run_cfg.yml
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
  VPINN:
    PDE:
      Type: Burger
    variational_form: 1
    Training:
      boundary: lower
```
This will run the model using all the settings from the `VPINN_cfg.yml` file *except* for those given in this `run.yml`
file.

> **_Note_**: The model comes with plenty of example configuration files in the `cfgs` folder. These are
> *configuration sets*, complete sets of run configurations and evaluation routines designed to produce specific
> plots. These also demonstrate how to load datasets to run the models; see below for more details.

## Plotting
All data and configuration files are saved alongside each run. This makes reproducing the plots
easy, since you can always re-run a simulation from the configuration files provided. However, you can
also re-evaluate a simulation, adding or modfiying plots, without having to re-train the model.
To plot from the last run, simply call
```console
utopya eval VPINN
```
This will re-evaluate the *last* run in your output path, using the default plots
given in `VPINN_plots.yml`. To re-evaluate a specific run, call
```console
utopya eval VPINN path/to/output/folder
```
You can update the plot settings given in `VPINN_plots.yml`, but again, these are defaults that are best left
unchanged. Instead, create a new plots configuration, and evaluate the model using it by calling
```console
utopya eval VPINN --eval-cfg path/to/eval/cfg
```
Even more convenient are so-called *configuration* sets:

## Configuration sets
Configuration sets are bundled run and evaluation configuration sets. Take a look at the `models/cfgs` folder:
it contains a number of examples. To run and evaluate the model from one of these configuration sets, just call
```console
utopya run VPINN --cfg-set cfg_set_name
```
replacing `cfg_set_name` with the name of the configuration set. To add a new set, simply create a new
folder in the `cfgs` folder (or anywhere else, but then taking care to pass an absolute path
rather than just a name). Running the configuration set will produce plots. If you wish to re-evaluate a run (perhaps plotting different figures),
you do not need to re-run the model, since the data has already been generated. Simply call

```console
utopya eval HarrisWilson --cfg-set <name_of_cfg_set>
```

This will re-evaluate the *last model you ran*. You can re-evaluate any dataset, of course, by
providing the path to that dataset, as before:

```console
utopya eval HarrisWilson path/to/output/folder --cfg-set <name_of_cfg_set>
```
## Modifying the configuration
To control the simulation, modify the entries in your run configuration. There are
a number of settings you can adjust. Remember, the run configuration only requires those entries to
wish to change with respect to the default values.
### Grid settings
The domain is controlled from `space` entry:

```yaml
parameter_space:
  VPINN:
    space:
      x:
        size: 10
        extent: [-1, 1]
      y:
        size: 12
        extent: [0, 2]
```
You can pass up to three dimensions. The number of dimensions you pass will determine the grid dimensionality.
These settings are used for training and evaluating the test functions. Once training is
complete, the model will make a prediction. If you want the prediction resolution to be
different from the training resolution, or to make predictions on a different domain,
you can pass a `predictions_grid` entry:
```yaml
VPINN:
  predictions_grid:
    x:
      size: 100
```
The model will recursively update any entries from the `space` configuration and use these
to plot the model predictions. This can be useful when wishing to make high-resolution predictions from
a low-resolution training scheme, for instance.
### Function settings
#### PDE and external forcing
You can adjust the PDE to use in the ``PDE/type`` entry. The `function` key controls the
forcing to use. Select from one of the permissible entries given in the `VPINN_cfg.yml`. Make sure you choose a
function that is adjusted to your space configuration. If you wish to add a new function, include it in the `EXAMPLES`
dictionary in `function_defintions.py` and add the name to the list of permissible arguments in the
`VPINN_cfg.yml`. The `!param` tag is a flag that tells the model to [check the parameters passed are
valid](https://docs.utopia-project.org/html/usage/run/config-validation.html): this can prevent cryptic error messages and unexpected behaviour, including silent errors.
You can adjust the scalar parameters for each PDE in the corresponding entry; for example, set the
viscosity of the Burger's equation by setting
```console
PDE:
  Burgers:
    nu: 0.2
```
#### Test functions
Adjust the test function configuration using the `test_functions` entry. Make sure the
test functions have the same dimension as the space. Currently, only `Legendre`, `Chebyshev`, and
`Sine` are supported test functions.
#### Weight function
The test function weighting is controlled from `test_functions/weight_function` entry;
weighting can be either `uniform` (all test functions have weight 1), or `exponential`, meaning
the `kl`-th function has weight `2**{-(k+l)}`.

### Neural Net configuration

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

### Training settings
You can modify the training settings, such as the batch size or the training device, from the
`Training` entry of the config:

```yaml
Training:
  batch_size: 1
  boundary: lower
  learning_rate: 0.001
  boundary_loss_weight: 1
  variational_loss_weight: 1
  device: cpu
  num_threads: ~
```
The `device` entry sets the training device. The default here is the `cpu`; you can set it to any
supported pytorch training device (e.g. `cuda` for most GPUs). Make sure your platform is configured to support the selected device.

> **_Warning_**: On Apple Silicon, set the device to `mps` to enable GPU training, provided you have followed the corresponding
> installation instructions (see above). However, training on `mps` is currently in beta, and many functions do net
> have a gradient. You may therefore receive a runtime error of the kind
>
> ```RuntimeError: derivative for <function> is not implemented```
>
> Training on the GPU is still very much WIP. It is possible you will not see a performance increase, or even a
> performance decline, as the loss calculation is very intensive, and pytorch does seem to have implemented
> efficient eager execution on all GPUs.

`utopya` automatically parallelises multiple runs; the number of CPU cores available to do this
can be specified under `worker_managers/num_workers` on the root-level configuration (i.e. on the same level as
`parameter_space`). The `Training/num_threads` entry controls the number of threads *per model run* to be used during training.
If you thus set `num_workers` to 4 and `num_threads` to 3, you will in total be able to use 12 threads.

#### Selecting the training boundary
The `boundary` entry sets the section of the `boundary` to use for training. It can be a keyword (`lower`, `upper`, `left`,
`right`, `front`, `back`), an index range to select a specific section of the boundary, or a combination
of both. For instance, on a two-dimensional grid, you can do
```yaml
boundary:
  - left
  - lower
  - !slice [10, 15]
```
to select the and left and lower boundary, as well as a section of the right boundary. The ``!slice`` argument takes
arguments of the kind ``(start, stop, step)``, as is common in Python. You can combine these arguments at will.
Note that the indexing range of the boundary begins (for two-dimensional grids) in the lower left corner and traverses the
grid boundary anti-clockwise. For three dimensional grids, the index begins on the lower left corner of the
front panel, wraps anti-clockwise around the sides, then selects the upper, and finally the lower panel.

## Loading data
Once you have generated grid and test function data, you can reuse it to train the neural net repeatedly. This will
speed up computations enormously, and also allow sweep runs with different neural net configurations.
To load a dataset, pass the path to the directory containing the `.h5` file to load to the load configuration entry of the config:

```yaml
VPINN:
  load_data:
    data_dir: path/to/folder
```
This will load all the required data and train the model on it; it will however not copy the loaded data to the
new directory by default, in order to conserve disk space. If you want the test function and grid data stored alongside the
neural net data (e.g. for data analysis and plotting purposes), set ``copy_data`` to true:

```yaml
VPINN:
  load_data:
    data_dir: path/to/folder
    copy_data: True
```

To turn off data loading, set the ``data_dir`` entry to ``~`` (``None`` in yaml), or delete the
``load_data`` entry entirely.

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

## 🚧 Tests (WIP)
To run tests, invoke
```bash
pytest tests
```
from the main folder.
