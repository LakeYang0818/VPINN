from matplotlib import rcParams
import numpy as np
import time
import torch
from typing import Any, List, Sequence, Union
import yaml

# Local imports
from function_definitions import f, u
from Utils.Datatypes.Grid import construct_grid, Grid
from Utils.Datatypes.DataSet import DataSet
from Utils.test_functions import testfunc_grid_evaluation
from Utils.utils import integrate
import Utils.Plots as Plots
from Utils.utils import validate_cfg
from Utils.VPINN import VPINN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load config file
with open('config.yml', 'r') as file:
    cfg = yaml.safe_load(file)

# ......................................................................................................................

if __name__ == "__main__":

    # Validate the configuration to prevent cryptic errors
    validate_cfg(cfg)

    # Get space parameters from the config
    dim: int = cfg['space']['dimension']
    eq_type: str = cfg['PDE']['type']
    var_form: int = cfg['variational_form']
    grid_size: Union[int, Sequence[int]] = cfg['space']['grid_size']['x'] if dim == 1 else [
        cfg['space']['grid_size']['x'], cfg['space']['grid_size']['y']]
    grid_boundary: Sequence = cfg['space']['boundary']['x'] if dim == 1 else [cfg['space']['boundary']['x'],
                                                                              cfg['space']['boundary']['y']]

    # Get the neural net architecture from the config
    n_nodes: int = cfg['architecture']['nodes_per_layer']
    n_layers: int = cfg['architecture']['layers']
    architecture: Union[List[int], Any] = [dim] + [n_nodes] * (n_layers + 1) + [1]

    # Get PDE constants from the config
    PDE_constants: dict = {'Burger': cfg['PDE']['Burger']['nu'],
                           'Helmholtz': cfg['PDE']['Helmholtz']['k'],
                           'PorousMedium': cfg['PDE']['PorousMedium']['m']}

    # Get type and number of test functions in each dimension
    test_func_type: str = cfg['Test functions']['type']
    test_func_dim: Union[int, Sequence[int]] = cfg['Test functions']['N_test_functions']['x'] if dim == 1 else [
        cfg['Test functions']['N_test_functions']['x'], cfg['Test functions']['N_test_functions']['y']]
    n_test_funcs: int = test_func_dim if dim == 1 else test_func_dim[0] * test_func_dim[1]

    # Construct the grid
    print("Constructing grid ...")
    grid: Grid = construct_grid(dim=dim, boundary=grid_boundary, grid_size=grid_size,
                                as_tensor=True, requires_grad=False, requires_normals=(var_form >= 2))

    # Evaluate the test functions and any required derivatives on the grid interior
    print("Evaluating test functions on the grid interior ... ")
    test_func_vals: DataSet = testfunc_grid_evaluation(grid, test_func_dim,
                                                       d=0, where='interior', which=test_func_type)
    d1test_func_vals: DataSet = testfunc_grid_evaluation(grid, test_func_dim,
                                                         d=1, where='interior',
                                                         which=test_func_type) if var_form >= 1 else None
    d2test_func_vals: DataSet = testfunc_grid_evaluation(grid, test_func_dim,
                                                         d=2, where='interior',
                                                         which=test_func_type) if var_form >= 2 else None

    # Evaluate the test functions on the grid boundary
    d1test_func_vals_bd: DataSet = testfunc_grid_evaluation(grid, test_func_dim,
                                                            d=1, where='boundary',
                                                            which=test_func_type) if var_form >= 2 else None

    # The weight function for the test functions. Takes an index or a tuple of indices
    # TO DO: this should be done more carefully
    if cfg['Test functions']['weighting']:
        weight_function = lambda x: 2 ** (-x[0]) * 2 ** (-x[1]) if dim == 2 else 2 ** (-x)
    else:
        weight_function = lambda x: 1

    # Integrate the external function over the grid against all the test functions.
    # This will be used to calculate the variational loss; this step is costly and only needs to be done once.
    print("Integrating test functions ...")
    f_integrated: DataSet = DataSet(coords=test_func_vals.coords,
                                    data=[integrate(f(grid.interior), test_func_vals.data[i], domain_volume=grid.volume)
                                          for i in range(n_test_funcs)],
                                    as_tensor=True,
                                    requires_grad=False)

    # Instantiate the model class
    model: VPINN = VPINN(architecture, eq_type, var_form,
                         pde_constants=PDE_constants,
                         learning_rate=cfg['learning_rate'],
                         activation_func=torch.relu).to(device)

    # Turn on tracking for the grid interior, on which the variational loss is calculated
    grid.interior.requires_grad = True

    # Prepare the training data. The training data consists of the explicit solution of the function on the boundary.
    # For the Burger's and Porous medium equation, training data is the initial data given
    # on the lower temporal boundary.
    print("Generating training data ...")
    if eq_type in ['Burger']:
        training_data: DataSet = DataSet(coords=grid.lower_boundary, data=u(grid.lower_boundary), as_tensor=True,
                                         requires_grad=False)
    else:
        training_data: DataSet = DataSet(coords=grid.boundary, data=u(grid.boundary), as_tensor=True,
                                         requires_grad=False)

    # Train the model
    print("Commencing training ...")
    b_weight, v_weight = cfg['boundary_loss_weight'], cfg['variational_loss_weight']
    start_time = time.time()
    for it in range(cfg['N_iterations'] + 1):

        model.optimizer.zero_grad()

        # Calculate the loss
        loss_b = model.boundary_loss(training_data)
        loss_v = model.variational_loss(grid, f_integrated, test_func_vals, d1test_func_vals, d2test_func_vals,
                                        d1test_func_vals_bd, weight_function)
        loss = b_weight * loss_b + v_weight * loss_v
        loss.backward()

        # Adjust the model parameters
        model.optimizer.step()

        # Track loss values
        loss_glob, loss_b_glob, loss_v_glob = loss.item(), loss_b.item(), loss_v.item()

        if it % 10 == 0:
            model.update_loss_tracker(it, loss_glob, loss_b_glob, loss_v_glob)
        if it % 100 == 0:
            print(f"Iteration {it}: total loss: {loss_glob}, loss_b: {loss_b_glob}, loss_v: {loss_v_glob}")

        del loss

    print(f"Training completed in {np.around(time.time() - start_time, 3)} seconds.")

    # Plot the results
    print("Plotting ... ")
    rcParams.update(cfg['plots']['rcParams'])

    # Generate the plot grid, which may be finer than the training grid
    plot_res = cfg['plots']['plot_resolution']['x'] if dim == 1 else [cfg['plots']['plot_resolution']['x'],
                                                                      cfg['plots']['plot_resolution']['y']]
    plot_grid: Grid = construct_grid(dim=dim, boundary=grid_boundary, grid_size=plot_res,
                                     requires_grad=False)

    # Get the model predictions on the plotting grid. Turn off tracking for the prediction data.
    predictions = model.forward(plot_grid.data).detach()

    # Plot an animation of the predictions
    if grid.dim == 2 and cfg['plots']['plot_animation']:
        Plots.animate(plot_grid, predictions)

    # Plot predicted vs actual values
    Plots.plot_prediction(cfg, plot_grid, predictions, grid_shape=plot_res,
                          plot_info_box=cfg['plots']['plot_info_box'])

    # Plot loss over time
    Plots.plot_loss(model.loss_tracker)

    # Plot test functions
    # Plots.plot_test_functions(plot_grid, order=min(6, n_test_funcs), d=1, which=test_func_type)

    # Save the config
    Plots.write_config(cfg)

    print("Done.")
