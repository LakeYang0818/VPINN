import numpy as np
import tensorflow as tf
from typing import Union, Sequence, List, Any
import yaml
from matplotlib import rcParams

# Local imports
import Utils.utils as utils
import Utils.plots as plots
from Utils.data_types import DataGrid, DataSet, Grid
from Utils.functions import f, u
from Utils.test_functions import test_function
from Utils.VPINN import VPINN

# Load config file
with open('config.yml', 'r') as file:
    cfg = yaml.safe_load(file)

# Set random seeds
np.random.seed(cfg['seed'])
tf.random.set_seed(cfg['seed'])
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# ..............................................................................

if __name__ == "__main__":

    # Validate the configuration to prevent cryptic errors
    utils.validate_cfg(cfg)

    # Get space parameters from the config
    dim: int = cfg['space']['dimension']
    eq_type: str = cfg['PDE']['type']
    grid_size: Union[int, Sequence[int]] = cfg['space']['grid_size']
    n_quad: Union[int, Sequence[int]] = cfg['N_quad']

    # Get the neural net architecture from the config
    n_nodes: int = cfg['architecture']['nodes_per_layer']
    n_layers: int = cfg['architecture']['layers']
    architecture: Union[List[int], Any] = [dim] + [n_nodes] * n_layers + [1]

    # Get PDE constants from the config
    PDE_constants: dict = {'Helmholtz': cfg['PDE']['Helmholtz']['k'],
                           'Burger': cfg['PDE']['Burger']['nu']}

    # Get number of test functions used
    n_test_func: int = cfg['N_test_functions']

    # Construct the grid
    print("Constructing grid ...")
    grid: Grid = utils.construct_grid(dim, cfg['space']['boundary'], grid_size)

    # Collect the quadrature points: these are weights on the hypercube [-1, 1]^dim
    print("Collecting quadrature points ...")
    quadrature_data: DataGrid = utils.get_quadrature_data(dim, cfg['N_quad'])

    # Scale the quadrature points to each grid element and calculate the Jacobians of the coordinate
    # transforms for each grid element. This only needs to be done once.
    print("Rescaling the quadrature points to the grid elements ... ")
    quadrature_data_scaled, jacobians = utils.scale_quadrature_data(grid, quadrature_data)

    # Integrate the external function over the grid against all the test functions.
    # This will be used to calculate the variational loss.
    print("Integrating test functions ...")
    f_integrated: DataSet = DataSet(x=list(range(1, n_test_func + 1)),
                                    f=[utils.integrate_over_grid(
                                        f, lambda x: test_function(x, i), quadrature_data,
                                        quadrature_data_scaled, jacobians)
                                        for i in range(1, n_test_func + 1)])

    # Build the model and initialize the optimiser
    print("Instantiating the model ... ")
    model: VPINN = VPINN(f_integrated,
                         quadrature_data,
                         quadrature_data_scaled,
                         jacobians,
                         grid.boundary,
                         n_test_func,
                         input_dim=dim,
                         architecture=architecture,
                         loss_weight=cfg['loss_weight'],
                         learning_rate=cfg['learning_rate'],
                         var_form=cfg['variational_form'],
                         eq_type=eq_type,
                         pde_params=PDE_constants,
                         activation=tf.math.sin)

    print("Beginning training ... ")
    # Prepare the training data. The training data consists of the explicit solution of the function on the boundary
    training_data: DataSet = DataSet(x=grid.boundary,
                                     f=u(grid.boundary) if dim > 1 else [u(grid.boundary[0]), u(grid.boundary[-1])],
                                     as_tensor=True)

    # Train the model
    model.train(training_data, cfg['N_iterations'])

    # Plot the results
    print("Plotting ... ")
    rcParams.update(cfg['plots']['rcParams'])
    plot_grid: Grid = utils.construct_grid(dim, cfg['space']['boundary'], cfg['plots']['plot_points'])
    prediction = model.evaluate(tf.constant(plot_grid.data))

    # Predicted vs actual values
    plots.plot_prediction(plot_grid, prediction.numpy(), grid)

    # Quadrature data
    plots.plot_quadrature_data(quadrature_data)

    # Loss over time
    plots.plot_loss(model.loss_tracker)
    print("Done.")
