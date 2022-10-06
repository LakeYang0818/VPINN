#!/usr/bin/env python3
from os.path import dirname as up
import sys
import h5py as h5
import numpy as np
import ruamel.yaml as yaml
import torch
import time
from typing import Any, List, Sequence, Union
import xarray as xr

from dantro._import_tools import import_module_from_path
from dantro import logging
import coloredlogs

sys.path.append(up(__file__))
sys.path.append(up(up(__file__)))

base = import_module_from_path(mod_path=up(up(__file__)), mod_str='include')
this = import_module_from_path(mod_path=up(__file__), mod_str='model')

log = logging.getLogger(__name__)
coloredlogs.install(fmt='%(levelname)s %(message)s', level='INFO', logger=log)


# ----------------------------------------------------------------------------------------------------------------------
# -- Model implementation ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


class VPINN:

    def __init__(
            self,
            name: str,
            *,
            rng: np.random.Generator,
            h5group: h5.Group,
            neural_net: base.NeuralNet,
            write_every: int = 1,
            write_start: int = 1,
            write_time: bool = False,
            grid: xr.DataArray,
            training_data: xr.DataArray = None,
            f_integrated: xr.DataArray,
            test_func_values: xr.DataArray,
            d1test_func_values: xr.DataArray = None,
            d2test_func_values: xr.DataArray = None,
            d1test_func_values_boundary: xr.DataArray = None,
            **__,
    ):
        """Initialize the model instance with a previously constructed RNG and
        HDF5 group to write the output data to.

        Args:
            name (str): The name of this model instance
            rng (np.random.Generator): The shared RNG
            h5group (h5.Group): The output file group to write data to
            neural_net: The neural network
            write_every: write every iteration
            write_start: iteration at which to start writing
            write_time: whether to write out the training time into a dataset
            grid (xr.Dataset): the grid
            training_data (xr.DataArray): the training_data, consisting of the values of the
              explicit solution on the boundary
            f_integrated (xr.DataArray): dataset containing the values of the external forcing integrated against the test
              functions
            d1test_func_values (xr.DataArray, optional): dataset containing the values of the test function first
              derivatives
            d2test_func_values (xr.DataArray, optional): dataset containing the values of the test function second
              derivatives
            d1test_func_values_boundary (xr.DataArray, optional): dataset containing the values of the test function first
              derivatives evaluated on the boundary
            **__: other arguments are ignored
        """
        self._name = name
        self._time = 0
        self._h5group = h5group
        self._rng = rng

        self.neural_net = neural_net
        self.neural_net.optimizer.zero_grad()
        self.current_loss = torch.tensor(0.0)
        self.current_boundary_loss = torch.tensor(1.0)
        self.current_boundary_loss = torch.tensor(2.0)

        # --- Set up chunked dataset to store the state data in --------------------------------------------------------
        # Setup chunked dataset to store the state data in
        self._dset_loss = self._h5group.create_dataset(
            "loss",
            (0, 3, 1),
            maxshape=(None, 3, 1),
            chunks=True,
            compression=3,
        )
        self._dset_loss.attrs['dim_names'] = ['time', 'loss type', 'dim_name__3']
        self._dset_loss.attrs["coords_mode__time"] = "start_and_step"
        self._dset_loss.attrs["coords__time"] = [write_start, write_every]
        self._dset_loss.attrs["coords_mode__loss type"] = "values"
        self._dset_loss.attrs["coords__loss type"] = ['total loss', 'boundary loss', 'variational loss']

        if write_time:
            self.dset_time = self._h5group.create_dataset(
                "computation_time",
                (0, 1),
                maxshape=(None, 1),
                chunks=True,
                compression=3,
            )
            self.dset_time.attrs['dim_names'] = ['epoch', 'training_time']
            self.dset_time.attrs["coords_mode__epoch"] = "trivial"
            self.dset_time.attrs["coords_mode__training_time"] = "trivial"

        self._write_every = write_every
        self._write_start = write_start

        self.training_data = training_data
        self.f_integrated = f_integrated
        self.test_func_values = test_func_values
        self.d1test_func_vals = d1test_func_values
        self.d2test_func_vals = d2test_func_values
        self.d1test_func_vals_boundary = d1test_func_values_boundary

    def epoch(self, *, boundary_loss_weight: float = 1.0, variational_loss_weight: float = 1.0):

        """ Trains the model for a single epoch """

        # Reset the neural net optimizer
        self.neural_net.optimizer.zero_grad()

        # Calculate the loss
        # boundary_loss = model.boundary_loss(self.training_data)
        # variational_loss = model.variational_loss(self.grid, self.f_integrated, self.test_func_vals,
        #                                           self.d1test_func_vals, self.d2test_func_vals,
        #                                           self.d1test_func_vals_bd, )
        # loss = boundary_loss_weight * boundary_loss + variational_loss_weight * variational_loss
        # loss.backward()

        boundary_loss = torch.rand(1)
        variational_loss = torch.rand(1)
        loss = torch.rand(1)

        # Adjust the model parameters
        # self.neural_net.optimizer.step()

        # Track loss values
        self.current_loss = loss.clone().detach().cpu().numpy()
        self.current_boundary_loss = boundary_loss.clone().detach().cpu().numpy()
        self.current_variational_loss = variational_loss.clone().detach().cpu().numpy()

        # Write the data
        self.write_data()
        self._time += 1

    def write_data(self):
        """Write the current state (loss and parameter predictions) into the state dataset.

        In the case of HDF5 data writing that is used here, this requires to
        extend the dataset size prior to writing; this way, the newly written
        data is always in the last row of the dataset.
        """
        if self._time >= self._write_start and (self._time % self._write_every == 0):
            self._dset_loss.resize(self._dset_loss.shape[0] + 1, axis=0)
            self._dset_loss[-1, :] = [self.current_loss, self.current_boundary_loss, self.current_variational_loss]

if __name__ == "__main__":

    cfg_file_path = sys.argv[1]

    log.note("   Preparing model run ...")
    log.note(f"   Loading config file:\n        {cfg_file_path}")
    with open(cfg_file_path, "r") as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.Loader)
    model_name = cfg.get("root_model_name", "SIR")
    log.note(f"   Model name:  {model_name}")
    model_cfg = cfg[model_name]

    # Select the training device and number of threads to use
    device = model_cfg['Training'].pop('device', None)
    if device is None:
      device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    num_threads = model_cfg['Training'].pop('num_threads', None)
    if num_threads is not None:
      torch.set_num_threads(num_threads)
    log.info(f"   Using '{device}' as training device. Number of threads: {torch.get_num_threads()}")

    # Get the random number generator
    log.note("   Creating global RNG ...")
    rng = np.random.default_rng(cfg["seed"])
    np.random.seed(cfg['seed'])
    torch.random.manual_seed(cfg['seed'])

    log.note(f"   Creating output file at:\n        {cfg['output_path']}")
    h5file = h5.File(cfg["output_path"], mode="w")
    h5group = h5file.create_group(model_name)

    eq_type: str = model_cfg['PDE']['type']
    var_form: int = model_cfg['variational_form']

    # Get PDE constants from the config
    PDE_constants: dict = {'Burger': model_cfg['PDE']['Burger']['nu'],
                           'Helmholtz': model_cfg['PDE']['Helmholtz']['k'],
                           'PorousMedium': model_cfg['PDE']['PorousMedium']['m']}

    # Initialise the neural net
    log.info("   Initializing the neural net ...")
    net = base.NeuralNet(input_size=len(model_cfg['space']),
                         output_size=1,
                         eq_type=eq_type,
                         var_form=var_form,
                         pde_constants=PDE_constants,
                         **model_cfg['NeuralNet']).to(device)

    # Get the data: grid, test function data, and training data. This is loaded from a file,
    # if provided, else synthetically generated

    data: dict = this.get_data(model_cfg.pop('load_from_file', None),
                                         model_cfg['space'],
                                         model_cfg['test_functions'],
                                         forcing = this.Examples[model_cfg['PDE']['type']]['f'],
                                         var_form = model_cfg['variational_form'],
                                         h5file = h5file)

    # Initialise the model
    log.info(f"   Initialising the model '{model_name}' ...")
    model = VPINN(
        model_name, rng=rng, h5group=h5group, neural_net=net,
        write_every=cfg['write_every'], write_start=cfg['write_start'],
        **data
    )

    num_epochs = cfg["num_epochs"]
    log.info(f"   Now commencing training for {num_epochs} epochs ...")
    for i in range(num_epochs):
        model.epoch(boundary_loss_weight=model_cfg['Training']['boundary_loss_weight'],
                    variational_loss_weight=model_cfg['Training']['variational_loss_weight'])

        log.progress(f"   Completed epoch {i+1} / {num_epochs}; "
                     f"   current loss: {model.current_loss}")

    log.info("   Simulation run finished.")
    log.info("   Wrapping up ...")
    h5file.close()

    log.success("   All done.")


#
# # Local imports
# from function_definitions import f, u
# from Utils.Datatypes.Grid import construct_grid, Grid
# from Utils.Datatypes.DataSet import DataSet
# from Utils.test_functions import testfunc_grid_evaluation
# from Utils.utils import integrate
# import Utils.Plots as Plots
# from Utils.utils import validate_cfg
# from Utils.VPINN import VPINN



#
# if __name__ == "__main__":
#
#     # Validate the configuration to prevent cryptic errors
#     validate_cfg(cfg)
#
#     # Get space parameters from the config
#     dim: int = cfg['space']['dimension']
#     eq_type: str = cfg['PDE']['type']
#     var_form: int = cfg['variational_form']
#     grid_size: Union[int, Sequence[int]] = cfg['space']['grid_size']['x'] if dim == 1 else [
#         cfg['space']['grid_size']['x'], cfg['space']['grid_size']['y']]
#     grid_boundary: Sequence = cfg['space']['boundary']['x'] if dim == 1 else [cfg['space']['boundary']['x'],
#                                                                               cfg['space']['boundary']['y']]
#
#     # Get the neural net architecture from the config
#     n_nodes: int = cfg['architecture']['nodes_per_layer']
#     n_layers: int = cfg['architecture']['layers']
#     architecture: Union[List[int], Any] = [dim] + [n_nodes] * (n_layers + 1) + [1]
#
#     # Get PDE constants from the config
#     PDE_constants: dict = {'Burger': cfg['PDE']['Burger']['nu'],
#                            'Helmholtz': cfg['PDE']['Helmholtz']['k'],
#                            'PorousMedium': cfg['PDE']['PorousMedium']['m']}
#
#     # Get type and number of test functions in each dimension
#     test_func_type: str = cfg['Test functions']['type']
#     test_func_dim: Union[int, Sequence[int]] = cfg['Test functions']['N_test_functions']['x'] if dim == 1 else [
#         cfg['Test functions']['N_test_functions']['x'], cfg['Test functions']['N_test_functions']['y']]
#     n_test_funcs: int = test_func_dim if dim == 1 else test_func_dim[0] * test_func_dim[1]
#
#     # Construct the grid
#     print("Constructing grid ...")
#     grid: Grid = construct_grid(dim=dim, boundary=grid_boundary, grid_size=grid_size,
#                                 as_tensor=True, requires_grad=False, requires_normals=(var_form >= 2))
#
#     # Evaluate the test functions and any required derivatives on the grid interior
#     print("Evaluating test functions on the grid interior ... ")
#     test_func_vals: DataSet = testfunc_grid_evaluation(grid, test_func_dim,
#                                                        d=0, where='interior', which=test_func_type)
#     d1test_func_vals: DataSet = testfunc_grid_evaluation(grid, test_func_dim,
#                                                          d=1, where='interior',
#                                                          which=test_func_type) if var_form >= 1 else None
#     d2test_func_vals: DataSet = testfunc_grid_evaluation(grid, test_func_dim,
#                                                          d=2, where='interior',
#                                                          which=test_func_type) if var_form >= 2 else None
#
#     # Evaluate the test functions on the grid boundary
#     d1test_func_vals_bd: DataSet = testfunc_grid_evaluation(grid, test_func_dim,
#                                                             d=1, where='boundary',
#                                                             which=test_func_type) if var_form >= 2 else None
#
#     # The weight function for the test functions. Takes an index or a tuple of indices
#     # TO DO: this should be done more carefully
#     if cfg['Test functions']['weighting']:
#         weight_function = lambda x: 2 ** (-x[0]) * 2 ** (-x[1]) if dim == 2 else 2 ** (-x)
#     else:
#         weight_function = lambda x: 1
#
#     # Integrate the external function over the grid against all the test functions.
#     # This will be used to calculate the variational loss; this step is costly and only needs to be done once.
#     print("Integrating test functions ...")
#     f_integrated: DataSet = DataSet(coords=test_func_vals.coords,
#                                     data=[integrate(f(grid.interior), test_func_vals.data[i], domain_volume=grid.volume)
#                                           for i in range(n_test_funcs)],
#                                     as_tensor=True,
#                                     requires_grad=False)
#
#     # Instantiate the model class
#     model: VPINN = VPINN(architecture, eq_type, var_form,
#                          pde_constants=PDE_constants,
#                          learning_rate=cfg['learning_rate'],
#                          activation_func=torch.relu).to(device)
#
#     # Turn on tracking for the grid interior, on which the variational loss is calculated
#     grid.interior.requires_grad = True
#
#     # Prepare the training data. The training data consists of the explicit solution of the function on the boundary.
#     # For the Burgers equation, training data is the initial data given
#     # on the lower temporal boundary.
#     print("Generating training data ...")
#     if eq_type in ['Burger', 'PorousMedium']:
#         training_data: DataSet = DataSet(coords=grid.lower_boundary, data=u(grid.lower_boundary), as_tensor=True,
#                                          requires_grad=False)
#     else:
#         training_data: DataSet = DataSet(coords=grid.boundary, data=u(grid.boundary), as_tensor=True,
#                                          requires_grad=False)
#
#     # Train the model
#     print("Commencing training ...")
#     b_weight, v_weight = cfg['boundary_loss_weight'], cfg['variational_loss_weight']
#     start_time = time.time()
#     for it in range(cfg['N_iterations'] + 1):
#
#         model.optimizer.zero_grad()
#
#         # Calculate the loss
#         loss_b = model.boundary_loss(training_data)
#         loss_v = model.variational_loss(grid, f_integrated, test_func_vals, d1test_func_vals, d2test_func_vals,
#                                         d1test_func_vals_bd, weight_function)
#         loss = b_weight * loss_b + v_weight * loss_v
#         loss.backward()
#
#         # Adjust the model parameters
#         model.optimizer.step()
#
#         # Track loss values
#         loss_glob, loss_b_glob, loss_v_glob = loss.item(), loss_b.item(), loss_v.item()
#
#         if it % 10 == 0:
#             model.update_loss_tracker(it, loss_glob, loss_b_glob, loss_v_glob)
#         if it % 100 == 0:
#             print(f"Iteration {it}: total loss: {loss_glob}, loss_b: {loss_b_glob}, loss_v: {loss_v_glob}")
#
#         del loss
#
#     print(f"Training completed in {np.around(time.time() - start_time, 3)} seconds.")
#
#     # Plot the results
#     print("Plotting ... ")
#     rcParams.update(cfg['plots']['rcParams'])
#
#     # Generate the plot grid, which may be finer than the training grid
#     plot_res = cfg['plots']['plot_resolution']['x'] if dim == 1 else [cfg['plots']['plot_resolution']['x'],
#                                                                       cfg['plots']['plot_resolution']['y']]
#     plot_grid: Grid = construct_grid(dim=dim, boundary=grid_boundary, grid_size=plot_res,
#                                      requires_grad=False)
#
#     # Get the model predictions on the plotting grid. Turn off tracking for the prediction data.
#     predictions = model.forward(plot_grid.data).detach()
#
#     # Plot an animation of the predictions
#     if grid.dim == 2 and cfg['plots']['plot_animation']:
#         Plots.animate(plot_grid, predictions)
#
#     # Plot predicted vs actual values
#     Plots.plot_prediction(cfg, plot_grid, predictions, model.loss_tracker, grid_shape=plot_res,
#                           plot_info_box=cfg['plots']['plot_info_box'])
#
#     # Plot loss over time
#     Plots.plot_loss(model.loss_tracker, write_data=cfg['plots']['write_loss_data'])
#
#     # Plot test functions
#     # Plots.plot_test_functions(plot_grid, order=min(6, n_test_funcs), d=1, which=test_func_type)
#
#     # Save the config
#     Plots.write_config(cfg)
#
#     print("Done.")
