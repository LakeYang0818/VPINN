from matplotlib import rcParams
import torch
from typing import Any, List, Sequence, Union
import yaml

# Local imports
from function_definitions import f, u
from Utils.Types.Grid import construct_grid, Grid
from Utils.Types.DataSet import DataSet
from Utils.test_functions import evaluate_test_funcs
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
    PDE_constants: dict = {'Helmholtz': cfg['PDE']['Helmholtz']['k'],
                           'Burger': cfg['PDE']['Burger']['nu']}

    # Get number of test functions in each dimension
    test_func_dim: Union[int, Sequence[int]] = cfg['N_test_functions']['x'] if dim == 1 else [
        cfg['N_test_functions']['x'], cfg['N_test_functions']['y']]
    n_test_funcs: int = test_func_dim if dim == 1 else test_func_dim[0] * test_func_dim[1]

    # Construct the grid
    print("Constructing grid ...")
    grid: Grid = construct_grid(dim, grid_boundary, grid_size, as_tensor=True, requires_grad=False)

    # Evaluate the test functions on grid points.
    print("Evaluating test functions on the grid interior ... ")
    test_func_vals, idx = evaluate_test_funcs(grid, test_func_dim)
    d1test_func_vals = evaluate_test_funcs(grid, test_func_dim, d=1, output_dim=2)[0] if var_form == 1 else None
    d2test_func_vals = evaluate_test_funcs(grid, test_func_dim, d=2, output_dim=2)[0] if var_form == 2 else None

    # Integrate the external function over the grid against all the test functions.
    # This will be used to calculate the variational loss; this step is costly and only needs to be done once.
    print("Integrating test functions ...")
    f_integrated: DataSet = DataSet(coords=idx,
                                    data=[integrate(f(grid.interior), test_func_vals[i], grid.volume)
                                          for i in range(n_test_funcs)], as_tensor=True, requires_grad=False)

    # Instantiate the model class
    model: VPINN = VPINN(architecture, eq_type, var_form,
                         pde_constants=PDE_constants,
                         learning_rate=cfg['learning_rate'],
                         activation_func=torch.tanh).to(device)

    # Turn on tracking for the grid interior, on which the variational loss is calculated
    grid.boundary.requires_grad = True
    grid.interior.requires_grad = True

    # Prepare the training data. The training data consists of the explicit solution of the function on the boundary.
    # For the Burger's equation, initial data is only given on the lower spacial boundary.
    print("Generating training data ...")
    if eq_type == 'Burger':
        lower_boundary = torch.reshape(torch.tensor(list(zip(grid.x, torch.zeros(len(grid.x))))), (len(grid.x), 2))
        training_data: DataSet = DataSet(coords=lower_boundary, data=u(lower_boundary), as_tensor=True,
                                         requires_grad=False)
    else:
        training_data: DataSet = DataSet(coords=grid.boundary, data=u(grid.boundary), as_tensor=True, requires_grad=False)

    # Train the model
    print("Commencing training ...")
    b_weight, v_weight = cfg['boundary_loss_weight'], cfg['variational_loss_weight']

    for it in range(cfg['N_iterations'] + 1):

        model.optimizer.zero_grad()

        # Calculate the loss
        loss_b = model.boundary_loss(training_data)
        loss_v = model.variational_loss(grid, f_integrated, test_func_vals, d1test_func_vals, d2test_func_vals)
        loss = v_weight * loss_v + b_weight * loss_b
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

    # Plot the results
    print("Plotting ... ")
    rcParams.update(cfg['plots']['rcParams'])

    # Generate the plot grid, which may be finer than the training grid
    plot_res = cfg['plots']['plot_resolution']['x'] if dim == 1 else [cfg['plots']['plot_resolution']['x'],
                                                                      cfg['plots']['plot_resolution']['y']]
    plot_grid: Grid = construct_grid(dim, grid_boundary, plot_res, requires_grad=False)

    # Get the model predictions on the plotting grid. Turn off tracking for the prediction data.
    predictions = model.forward(plot_grid.data).detach()

    # Plot an animation of the predictions
    if grid.dim == 2 and cfg['plots']['plot_animation']:
        Plots.animate(plot_grid, predictions)

    # Plot predicted vs actual values
    Plots.plot_prediction(cfg, plot_grid, predictions, grid_shape=plot_res)

    # Plot loss over time
    Plots.plot_loss(model.loss_tracker)

    # Plot test functions
    Plots.plot_test_functions(plot_grid, order=min(4, n_test_funcs), d=0)

    # Save the config
    Plots.write_config(cfg)

    print("Done.")
