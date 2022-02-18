import numpy as np
import tensorflow as tf
from typing import Union, Sequence, List, Any
import yaml
import time
# Local imports
from Utils.functions import u, f
from Utils.test_functions import test_function
from Utils.data_types import DataSet, DataGrid, Grid
# from Utils.plotting import plot_train_quad_pts, plot_loss, plot_prediction, plot_error
from Utils.VPINN import VPINN
import Utils.utils as utils

# Load config file
with open('config.yml', 'r') as file:
    cfg = yaml.safe_load(file)

# Set random seeds
np.random.seed(cfg['seed'])
tf.random.set_seed(cfg['seed'])

# ..............................................................................
if __name__ == "__main__":

    # Get space parameters from the config
    dim: int = cfg['space']['dimension']
    grid_size: Union[int, Sequence[int]] = cfg['space']['grid_size']
    boundary: Sequence[Union[Sequence, int]] = cfg['space']['boundary']

    if dim == 1 and grid_size == 3:
        print("Grid size should be at least 4! Increasing grid size.")
        grid_size = 4

    if dim not in {1, 2}:
        raise ValueError(f'Argument {dim} not valid! Dimension must be either 1 or 2!')

    # Get the neural net architecture
    n_nodes = cfg['architecture']['nodes_per_layer']
    n_layers = cfg['architecture']['layers']
    architecture: Union[List[int], Any] = [dim] + [n_nodes] * n_layers + [1]

    # Get variational form number and equation type
    var_eq_type = (cfg['variational_form'], cfg['PDE']['type'])

    # Get number of test functions used
    n_test_func: int = cfg['N_test_functions']

    # Construct the grid and the quadrature points
    print("Constructing grid ...")
    grid: Grid = utils.construct_grid(dim, boundary, grid_size)

    print("Collecting quadrature points ...")
    quadrature_data: DataGrid = utils.get_quadrature_data(dim, cfg['N_quad'])

    # Integrate the external function against all the test functions
    # and over every grid element. This will be used to calculate the variational loss
    print("Integrating test functions ...")
    test_func_vals = []
    for n in range(1, n_test_func + 1):
        test_func_vals.append(test_function(quadrature_data.grid, n))

    # This can be a dataset containing the values of f on the domain for each test function!!!
    f_integrated: list[list[float]] = utils.integrate_f_over_grid(f, grid, quadrature_data, test_func_vals, dim)

    # Build the model and initialize the optimiser
    model: VPINN = VPINN(f_integrated,
                         input_dim=dim,
                         architecture=architecture,
                         loss_weight=cfg['loss_weight'],
                         learning_rate=cfg['learning_rate'],
                         var_form=cfg['variational_form'],
                         eq_type=cfg['PDE']['type'],
                         activation='swish')

    # Prepare the training data. The training data consists of the explicit solution of the function on the boundary
    training_data: DataSet = DataSet(x=grid.boundary, f=u(grid.boundary), as_tensor = True)

    # Train the model
    for epoch in range(cfg['N_iterations']):
        print(f'Start of epoch {epoch}')
        start_time = time.time()

        for step, (x_train, y_train) in enumerate(training_data):
            loss = model.train(x_train, y_train, grid = grid, quads = quadrature_data, n_test_functions = n_test_func)

        # Log every batch number
            if step % cfg['batch_size'] == 0:
                print(f"Training loss (for one batch) at step {step}: {loss.numpy()[0]:.4f}.")

        train_acc = model.train_acc_metric.result()
        print(f"Training acc over epoch: {float(train_acc)}")

        # Reset training metrics at the end of each epoch
        model.train_acc_metric.reset_states()

        print(f"Time taken: {(time.time()-start_time):.2f}")

    # random_points: Sequence = utils.get_random_points(grid, n_points=cfg["N_f"])
    # training_data_interior: DataSet = DataSet(x=utils.get_random_points(grid, n_points=cfg["N_f"]), f=f(random_points))
    #
    # # The test data is the exact solution evaluated on the grid. It is used for plotting.
    # test_data: DataSet = DataSet(x=grid.data, f=u(grid.data))


    # # Plot the results
    # plot_train_quad_pts(x_quad_train, x_train, x_f_train)
    # plot_loss(errors)
    # plot_prediction(x_test, u_test, u_pred)
    # plot_error(x_test, u_test, u_pred)