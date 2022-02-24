import numpy as np
import os
import torch
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
    # print("Constructing grid ...")
    # grid: Grid = utils.construct_grid(dim, cfg['space']['boundary'], grid_size)

    # Collect the quadrature points: these are weights on the hypercube [-1, 1]^dim
    # print("Collecting quadrature points ...")
    # quadrature_data: DataGrid = utils.get_quadrature_data(dim, cfg['N_quad'])

    # Scale the quadrature points to each grid element and calculate the Jacobians of the coordinate
    # transforms for each grid element. This only needs to be done once.
    # print("Rescaling the quadrature points to the grid elements ... ")
    # quadrature_data_scaled, jacobians = utils.scale_quadrature_data(grid, quadrature_data)

    # Integrate the external function over the grid against all the test functions.
    # This will be used to calculate the variational loss and only needs to be done once.
    # print("Integrating test functions ...")
    # f_integrated: DataSet = DataSet(x=[i for i in range(0, n_test_func)],
    #                                 f=[utils.integrate_over_grid(f, lambda x: test_function(x, i), quadrature_data,
    #                                                              quadrature_data_scaled, jacobians, as_tensor=True) for
    #                                    i in range(1, n_test_func + 1)])

    # Integrate the test functions against all quadrature points on the grid
    # print("Evaluating test functions on all quadrature points on the grid ... ")
    # test_funcs_eval: DataSet = DataSet(x=[i for i in range(n_test_func)],
    #                                    f=[[test_function(quadrature_data_scaled[j].grid.data, i) for j in
    #                                        range(len(quadrature_data_scaled))][0]
    #                                       for i in range(1, n_test_func + 1)])

    # Prepare the training data. The training data consists of the explicit solution of the function on the boundary
    # training_data: DataSet = DataSet(x=grid.boundary,
    #                                  f=u(grid.boundary) if dim > 1 else [u(grid.boundary[0]), u(grid.boundary[-1])])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model: VPINN = VPINN(architecture=architecture).to(device)
    print(model)
    print(model.forward(torch.tensor(0.4)))

    # # Instantiate the model class
    # print("Instantiating the model ... ")
    # model: VPINN = VPINN(f_integrated,
    #                      quadrature_data,
    #                      quadrature_data_scaled,
    #                      jacobians,
    #                      grid.boundary,
    #                      training_data,
    #                      test_funcs_eval,
    #                      input_dim=dim,
    #                      architecture=architecture,
    #                      loss_weight=cfg['loss_weight'],
    #                      learning_rate=cfg['learning_rate'],
    #                      var_form=cfg['variational_form'],
    #                      eq_type=eq_type,
    #                      pde_params=PDE_constants,
    #                      activation=tf.math.sin)
    #
    # # Train the model
    # print("Beginning training ... ")
    # model.train(cfg['N_iterations'])
    #
    # # Plot the results
    # print("Plotting ... ")
    # rcParams.update(cfg['plots']['rcParams'])
    # plot_grid: Grid = utils.construct_grid(dim, cfg['space']['boundary'], cfg['plots']['plot_points'])
    # prediction = model.evaluate(tf.constant(plot_grid.data))
    #
    # # Predicted vs actual values
    # plots.plot_prediction(plot_grid, prediction.numpy(), grid)
    #
    # # Quadrature data
    # plots.plot_quadrature_data(quadrature_data)
    #
    # # Loss over time
    # plots.plot_loss(model.loss_tracker)
    #
    # # plots.plot_test_functions(plot_grid, n_test_func)
    #
    # print("Done.")
