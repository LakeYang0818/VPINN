from matplotlib import rcParams
import numpy as np
import torch
from typing import Union, Sequence, List, Any
import yaml

# Local imports
import Utils.utils as utils
import Utils.plots as plots
from Utils.data_types import DataSet, Grid
from Utils.functions import f, u
from Utils.test_functions import test_function
from Utils.VPINN import VPINN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load config file
with open('config.yml', 'r') as file:
    cfg = yaml.safe_load(file)

# ..............................................................................

if __name__ == "__main__":

    # Validate the configuration to prevent cryptic errors
    utils.validate_cfg(cfg)

    # Get space parameters from the config
    dim: int = cfg['space']['dimension']
    eq_type: str = cfg['PDE']['type']
    grid_size: Union[int, Sequence[int]] = cfg['space']['grid_size']

    # Get the neural net architecture from the config
    n_nodes: int = cfg['architecture']['nodes_per_layer']
    n_layers: int = cfg['architecture']['layers']
    architecture: Union[List[int], Any] = [dim] + [n_nodes] * (n_layers+1) + [1]

    # Get PDE constants from the config
    PDE_constants: dict = {'Helmholtz': cfg['PDE']['Helmholtz']['k'],
                           'Burger': cfg['PDE']['Burger']['nu']}

    # Get number of test functions used
    n_test_func: int = cfg['N_test_functions']

    # Construct the grid
    print("Constructing grid ...")
    grid: Grid = utils.construct_grid(dim, cfg['space']['boundary'], grid_size, as_tensor=True)

    # Integrate the external function over the grid against all the test functions.
    # This will be used to calculate the variational loss and only needs to be done once.
    print("Integrating test functions ...")
    f_integrated: DataSet = DataSet(x=[i for i in range(0, n_test_func)],
                                    data=[utils.integrate(f, lambda x: test_function(x, i), grid.interior,
                                                          as_tensor=True)
                                          for i in range(1, n_test_func + 1)], as_tensor=True)

    # Evaluate the test functions against points on the grid
    print("Evaluating test functions on the grid interior ... ")
    test_func_vals: DataSet = DataSet(x=[i for i in range(n_test_func)],
                                       data=[test_function(grid.interior, i) for i in range(1, n_test_func + 1)],
                                       as_tensor=True, requires_grad=False)

    # Prepare the training data. The training data consists of the explicit solution of the function on the boundary
    training_data: DataSet = DataSet(x=grid.boundary,
                                     data=u(grid.boundary) if dim > 1
                                     else torch.stack([u(grid.boundary[0]), u(grid.boundary[-1])]))

    # Now turn on the tracking for the grid
    grid.data.requires_grad=True
    grid.interior.requires_grad=True
    grid.boundary.requires_grad=True

    # Instantiate the model class
    model: VPINN = VPINN(architecture, torch.sin, cfg['loss_weight']).to(device)

    # Train the model
    model.train_custom(training_data, f_integrated, test_func_vals, grid, cfg['N_test_functions'], cfg['N_iterations'])

    print("Done")

    # Plot the results
    print("Plotting ... ")
    rcParams.update(cfg['plots']['rcParams'])
    plot_grid: Grid = utils.construct_grid(dim, cfg['space']['boundary'], cfg['plots']['plot_grid'])

    # Get the model predictions on the plotting grid. Turn off tracking for the prediction data.
    predictions = model.forward(plot_grid.data).detach()

    # Plot predicted vs actual values
    plots.plot_prediction(plot_grid, predictions, grid_shape=cfg['plots']['plot_grid'])

    # Plot loss over time
    plots.plot_loss(model.loss_tracker)