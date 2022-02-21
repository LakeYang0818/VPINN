import numpy as np
import tensorflow as tf
import time
from typing import Union, Sequence, List, Any
import yaml

# Local imports
import Utils.utils as utils
from Utils.data_types import DataGrid, DataSet, Grid
from Utils.functions import f, u
from Utils.test_functions import test_function
# from Utils.plotting import plot_train_quad_pts, plot_loss, plot_prediction, plot_error
from Utils.VPINN import VPINN

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
    eq_type: str = cfg['PDE']['type']
    grid_size: Union[int, Sequence[int]] = cfg['space']['grid_size']
    n_quad: Union[int, Sequence[int]] = cfg['N_quad']

    # Check configuration settings are valid to prevent cryptic errors
    if dim not in {1, 2}:
        raise ValueError(f'Argument {dim} not supported! Dimension must be either 1 or 2!')
    if dim == 1:
        if eq_type == 'Burger':
            raise TypeError('The Burgers equation requires a two-dimensional grid!')
        if isinstance(grid_size, Sequence):
            raise TypeError('The grid size should be a scalar! Adjust the config.')
        if isinstance(n_quad, Sequence):
            raise TypeError('The number of quadrature points should be a scalar! Adjust the config.')
    else:
        if not isinstance(grid_size, Sequence):
            raise TypeError('The grid size must be a sequence of the form [n_y, n_x]! '
                            'Adjust the config.')
        if not isinstance(n_quad, Sequence):
            raise TypeError('The number of quadrature points must be a sequence of the form [n_y, n_x]! '
                            'Adjust the config.')
    if 3 in grid_size:
        print("Grid size should be at least 4! Increasing grid size.")
        if isinstance(grid_size, Sequence):
            grid_size = [x if x != 3 else 4 for x in grid_size]
        else:
            grid_size = 4

    # Get the neural net architecture
    n_nodes: int = cfg['architecture']['nodes_per_layer']
    n_layers: int = cfg['architecture']['layers']
    architecture: Union[List[int], Any] = [dim] + [n_nodes] * n_layers + [1]

    # Get PDE constants
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
    model: VPINN = VPINN(f_integrated,
                         input_dim=dim,
                         architecture=architecture,
                         loss_weight=cfg['loss_weight'],
                         learning_rate=cfg['learning_rate'],
                         var_form=cfg['variational_form'],
                         eq_type=eq_type,
                         activation='relu')

    # Prepare the training data. The training data consists of the explicit solution of the function on the boundary
    training_data: DataSet = DataSet(x=grid.boundary,
                                     f=u(grid.boundary) if dim > 1 else [u(grid.boundary[0]), u(grid.boundary[-1])],
                                     as_tensor=True)

    # Train the model
    for epoch in range(cfg['N_iterations']):
        print(f'Start of epoch {epoch}')
        start_time = time.time()

        for step, (x_train, y_train) in enumerate(training_data):
            loss = model.train(x_train, y_train,
                               quads=quadrature_data,
                               quads_scaled=quadrature_data_scaled,
                               jacobians=jacobians,
                               grid_boundary=grid.boundary,
                               n_test_functions=n_test_func,
                               pde_params=PDE_constants)

            # Log every batch number
            if step % cfg['batch_size'] == 0:
                print(f"Training loss (for one batch) at step {step}: {loss.numpy()[0]:.4f}.")

        print(f"Time taken: {(time.time() - start_time):.2f}")

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
