from matplotlib import rcParams
import torch
from typing import Any, List, Sequence, Union
import yaml

# Local imports
from Utils.Types.Grid import construct_grid, Grid
from Utils.Types.DataSet import DataSet
from Utils.functions import f, integrate, u, test_function
import Utils.plots as plots
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
    grid: Grid = construct_grid(dim, cfg['space']['boundary'], grid_size, as_tensor=True, requires_grad=False)

    # Evaluate the test functions on grid points. This only needs to be done once.
    print("Evaluating test functions on the grid interior ... ")
    test_func_vals: DataSet = DataSet(coords=[[i] for i in range(n_test_func)],
                                      data=[test_function(grid.interior, i) for i in range(1, n_test_func + 1)],
                                      as_tensor=True, requires_grad=False)

    # Integrate the external function over the grid against all the test functions.
    # This will be used to calculate the variational loss and only needs to be done once.
    print("Integrating test functions ...")
    f_integrated: DataSet = DataSet(coords=[[i] for i in range(n_test_func)],
                                    data=[integrate(f(grid.interior), test_func_vals.data[i])
                                          for i in range(n_test_func)],
                                    as_tensor=True, requires_grad=False)

    # Instantiate the model class
    model: VPINN = VPINN(architecture, eq_type, var_form,
                         pde_constants=PDE_constants,
                         learning_rate=cfg['learning_rate'],
                         activation_func=torch.sin).to(device)

    # Turn on tracking for the grid
    # Note: this actually only needs to happen for the domain of integration (in our case the interior)
    grid.data.requires_grad = True
    grid.interior.requires_grad = True
    grid.boundary.requires_grad = True

    # Prepare the training data. The training data consists of the explicit solution of the function on the boundary
    print("Generating training data ...")
    training_data: DataSet = DataSet(coords=grid.boundary, data=u(grid.boundary), as_tensor=True, requires_grad=False)

    # Train the model
    print("Commencing training ...")
    loss_w: float = cfg['loss_weight']
    for it in range(cfg['N_iterations']):

        model.optimizer.zero_grad()

        # Calculate the loss
        loss_b = model.boundary_loss(training_data)
        loss_v = model.variational_loss(grid, f_integrated, test_func_vals)
        loss = loss_b + loss_w * loss_v
        loss.backward(retain_graph=True)

        # Adjust the model parameters
        model.optimizer.step()

        # Track loss values
        loss_glob, loss_b_glob, loss_v_glob = loss.item(), loss_b.item(), loss_v.item()

        if it % 10 == 0:
            model.update_loss_tracker(it, loss_glob, loss_b_glob, loss_v_glob)
        if it % 100 == 0:
            print(f"Iteration {it}: total loss: {loss_glob}, loss_b: {loss_b_glob}, loss_v: {loss_v_glob}")

    # Plot the results
    print("Plotting ... ")
    rcParams.update(cfg['plots']['rcParams'])

    # Generate the plot grid, which may be finer than the training grid
    plot_grid: Grid = construct_grid(dim, cfg['space']['boundary'], cfg['plots']['plot_grid'], requires_grad=False)

    # Get the model predictions on the plotting grid. Turn off tracking for the prediction data.
    predictions = model.forward(plot_grid.data).detach()

    # Plot predicted vs actual values
    plots.plot_prediction(plot_grid, predictions, grid_shape=cfg['plots']['plot_grid'])

    # Plot loss over time
    plots.plot_loss(model.loss_tracker)

    print("Done.")
