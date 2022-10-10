#!/usr/bin/env python3
import sys
import time
from itertools import chain
from os.path import dirname as up
from typing import Union

import coloredlogs
import h5py as h5
import numpy as np
import ruamel.yaml as yaml
import torch
import xarray as xr
from dantro import logging
from dantro._import_tools import import_module_from_path
from paramspace.tools import recursive_update

sys.path.append(up(__file__))
sys.path.append(up(up(__file__)))

base = import_module_from_path(mod_path=up(up(__file__)), mod_str="include")
this = import_module_from_path(mod_path=up(__file__), mod_str="model")

log = logging.getLogger(__name__)
coloredlogs.install(fmt="%(levelname)s %(message)s", level="INFO", logger=log)


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
        device: str,
        write_every: int = 1,
        write_start: int = 1,
        grid: xr.DataArray,
        training_data: xr.Dataset = None,
        f_integrated: xr.DataArray,
        test_func_values: xr.DataArray,
        d1test_func_values: xr.DataArray = None,
        d2test_func_values: xr.DataArray = None,
        d1test_func_values_boundary: xr.DataArray = None,
        weight_function: callable = lambda x: 1,
        **__,
    ):
        """Initialize the model instance with a previously constructed RNG and
        HDF5 group to write the output data to.

        Args:
            name (str): The name of this model instance
            rng (np.random.Generator): The shared RNG
            h5group (h5.Group): The output file group to write data to
            neural_net: The neural network
            device: the device to use
            write_every: write every iteration
            write_start: iteration at which to start writing
            grid (xr.DataArray): the grid
            boundary (xr.Dataset): the grid boundary
            training_data (xr.Datatset): the training_data, consisting of the values of the
              explicit solution on the boundary
            f_integrated (xr.DataArray): dataset containing the values of the external forcing integrated against the test
              functions
            d1test_func_values (xr.DataArray, optional): dataset containing the values of the test function first
              derivatives
            d2test_func_values (xr.DataArray, optional): dataset containing the values of the test function second
              derivatives
            d1test_func_values_boundary (xr.DataArray, optional): dataset containing the values of the test function first
              derivatives evaluated on the boundary
            weight_function (callable, optional): a function to use to weight the test functions
            **__: other arguments are ignored
        """

        def _tf_to_tensor(test_funcs: xr.DataArray) -> Union[None, torch.Tensor]:

            """Unpacks a DataArray of test function values and returns a stacked torch.Tensor"""

            if test_funcs is None:
                return None

            return torch.reshape(
                torch.from_numpy(
                    test_funcs.isel(
                        {
                            var: slice(1, -1)
                            for var in test_funcs.attrs["space_dimensions"]
                        }
                    ).data
                ),
                (len(test_funcs.coords["tf_idx"]), -1, 1),
            ).float()

        def _dtf_to_tensor(test_funcs: xr.DataArray) -> Union[None, torch.Tensor]:

            """Unpacks a DataArray of test function derivatives and returns a stacked torch.Tensor"""

            if test_funcs is None:
                return None

            return torch.reshape(
                torch.from_numpy(
                    test_funcs.isel(
                        {
                            var: slice(1, -1)
                            for var in test_funcs.attrs["space_dimensions"]
                        }
                    ).data
                ),
                (
                    len(test_funcs.coords["tf_idx"]),
                    -1,
                    test_funcs.attrs["grid_dimension"],
                ),
            ).float()

        self._name = name
        self._time = 0
        self._h5group = h5group
        self._rng = rng

        self.neural_net = neural_net
        self.neural_net.optimizer.zero_grad()
        self.current_loss = torch.tensor(0.0)
        self.current_boundary_loss = torch.tensor(0.0)
        self.current_boundary_loss = torch.tensor(0.0)

        # --- Set up chunked dataset to store the state data in --------------------------------------------------------
        # Setup chunked dataset to store the state data in
        self._dset_loss = self._h5group.create_dataset(
            "loss",
            (0, 3, 1),
            maxshape=(None, 3, 1),
            chunks=True,
            compression=3,
        )
        self._dset_loss.attrs["dim_names"] = ["time", "loss type", "dim_name__3"]
        self._dset_loss.attrs["coords_mode__time"] = "start_and_step"
        self._dset_loss.attrs["coords__time"] = [write_start, write_every]
        self._dset_loss.attrs["coords_mode__loss type"] = "values"
        self._dset_loss.attrs["coords__loss type"] = [
            "total loss",
            "boundary loss",
            "variational loss",
        ]

        # Write the training time after each epoch
        self.dset_time = self._h5group.create_dataset(
            "computation_time",
            (0, 1),
            maxshape=(None, 1),
            chunks=True,
            compression=3,
        )
        self.dset_time.attrs["dim_names"] = ["epoch", "dim_name__1"]
        self.dset_time.attrs["coords_mode__epoch"] = "trivial"

        self._write_every = write_every
        self._write_start = write_start

        # The grid interior
        self.grid: torch.Tensor = (
            torch.reshape(
                torch.from_numpy(
                    grid.isel(
                        {val: slice(1, -1) for val in grid.attrs["space_dimensions"]}
                    ).to_numpy()
                ).float(),
                (-1, grid.attrs["grid_dimension"]),
            )
            .requires_grad_(True)
            .to(device)
        )

        # The grid boundary
        self.grid_boundary: torch.Tensor = (
            torch.from_numpy(
                training_data.sel(
                    variable=grid.attrs["space_dimensions"], drop=True
                ).data.to_numpy()
            )
            .float()
            .to(device)
        )

        # The grid normals
        self.grid_normals: torch.Tensor = (
            torch.from_numpy(training_data.sel(variable=["n"]).data.to_numpy())
            .float()
            .to(device)
        )

        # The density of the grid
        self.domain_density = grid.attrs["grid_density"]

        # Training data (boundary conditions)
        self.training_data: torch.Tensor = (
            torch.from_numpy(
                training_data.sel(variable=["u"], drop=True).data.to_numpy()
            )
            .float()
            .to(device)
        )

        # Value of the external function integrated against all the test functions
        self.f_integrated: torch.Tensor = torch.reshape(
            torch.from_numpy(f_integrated.data), (-1, 1)
        )

        # Test function values on the grid interior, indexed by their (multi-)index and grid coordinate
        self.test_func_values: torch.Tensor = _tf_to_tensor(test_func_values)

        self.d1test_func_values: Union[None, torch.Tensor] = _dtf_to_tensor(
            d1test_func_values
        )

        self.d2test_func_values: Union[None, xr.DataArray] = _dtf_to_tensor(
            d2test_func_values
        )
        self.d1test_func_values_boundary: Union[
            None, xr.DataArray
        ] = d1test_func_values_boundary.to_array()

        self.weights = torch.stack(
            [
                weight_function(np.array(idx))
                for idx in test_func_values.coords["tf_idx"].data
            ]
        )

    def epoch(
        self, *, boundary_loss_weight: float = 1.0, variational_loss_weight: float = 1.0
    ):

        """Trains the model for a single epoch"""

        start_time = time.time()

        # Reset the neural net optimizer
        self.neural_net.optimizer.zero_grad()

        # Calculate the boundary loss
        boundary_loss = torch.nn.functional.mse_loss(
            self.neural_net.forward(self.grid_boundary), self.training_data
        )

        variational_loss = self.neural_net.variational_loss(
            self.grid,
            self.grid_boundary,
            self.grid_normals,
            self.f_integrated,
            self.test_func_values,
            self.weights,
            self.domain_density,
            self.d1test_func_values,
            self.d2test_func_values,
            self.d1test_func_values_boundary,
        )

        loss = (
            boundary_loss_weight * boundary_loss
            + variational_loss_weight * variational_loss
        )

        loss.backward()

        # Adjust the model parameters
        self.neural_net.optimizer.step()

        # Track loss values
        self.current_loss = loss.clone().detach().cpu().numpy()
        self.current_boundary_loss = boundary_loss.clone().detach().cpu().numpy()
        self.current_variational_loss = variational_loss.clone().detach().cpu().numpy()

        # Write the data
        self.write_data()
        self._time += 1

        # Write the training time (wall clock time)
        self.dset_time.resize(self.dset_time.shape[0] + 1, axis=0)
        self.dset_time[-1, :] = time.time() - start_time

    def write_data(self):
        """Write the current state (loss and parameter predictions) into the state dataset.

        In the case of HDF5 data writing that is used here, this requires to
        extend the dataset size prior to writing; this way, the newly written
        data is always in the last row of the dataset.
        """
        if self._time >= self._write_start and (self._time % self._write_every == 0):
            self._dset_loss.resize(self._dset_loss.shape[0] + 1, axis=0)
            self._dset_loss[-1, :] = [
                [self.current_loss],
                [self.current_boundary_loss],
                [self.current_variational_loss],
            ]


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
    device = model_cfg["Training"].get("device", None)
    if device is None:
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    num_threads = model_cfg["Training"].get("num_threads", None)
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    log.info(
        f"   Using '{device}' as training device. Number of threads: {torch.get_num_threads()}"
    )

    # Get the random number generator
    log.note("   Creating global RNG ...")
    rng = np.random.default_rng(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.random.manual_seed(cfg["seed"])

    log.note(f"   Creating output file at:\n        {cfg['output_path']}")
    h5file = h5.File(cfg["output_path"], mode="w")
    h5group = h5file.create_group(model_name)

    eq_type: str = model_cfg["PDE"]["type"]
    var_form: int = model_cfg["variational_form"]

    # Get PDE constants from the config
    PDE_constants: dict = {
        "Burger": model_cfg["PDE"]["Burger"]["nu"],
        "Helmholtz": model_cfg["PDE"]["Helmholtz"]["k"],
        "PorousMedium": model_cfg["PDE"]["PorousMedium"]["m"],
    }

    # Initialise the neural net
    log.info("   Initializing the neural net ...")
    net = base.NeuralNet(
        input_size=len(model_cfg["space"]),
        output_size=1,
        eq_type=eq_type,
        var_form=var_form,
        pde_constants=PDE_constants,
        **model_cfg["NeuralNet"],
    ).to(device)

    test_func_dict = model_cfg["test_functions"]

    # Get the data: grid, test function data, and training data. This is loaded from a file,
    # if provided, else synthetically generated
    data: dict = this.get_data(
        model_cfg.get("load_from_file", None),
        model_cfg["space"],
        test_func_dict,
        solution=this.EXAMPLES[model_cfg["PDE"]["function"]]["u"],
        forcing=this.EXAMPLES[model_cfg["PDE"]["function"]]["f"],
        var_form=model_cfg["variational_form"],
        boundary_isel=model_cfg["Training"].get("boundary", None),
        h5file=h5file,
    )

    # Initialise the model
    log.info(f"   Initialising the model '{model_name}' ...")
    model = VPINN(
        model_name,
        rng=rng,
        h5group=h5group,
        neural_net=net,
        device=device,
        write_every=cfg["write_every"],
        write_start=cfg["write_start"],
        weight_function=this.WEIGHT_FUNCTIONS[
            test_func_dict["weight_function"].lower()
        ],
        **data,
    )

    num_epochs = cfg["num_epochs"]
    log.info(f"   Now commencing training for {num_epochs} epochs ...")
    for _ in range(num_epochs):
        model.epoch(
            boundary_loss_weight=model_cfg["Training"]["boundary_loss_weight"],
            variational_loss_weight=model_cfg["Training"]["variational_loss_weight"],
        )

        if _ % 100 == 0:
            log.progress(
                f"   Completed epoch {_} / {num_epochs}; "
                f"   current loss: {model.current_loss}"
            )

    log.info("   Simulation run finished. Generating prediction ...")

    # Get the plot grid, which can be finer than the training grid, if specified
    plot_grid = base.construct_grid(
        recursive_update(model_cfg["space"], model_cfg.get("predictions_grid", {}))
    )

    predictions = xr.apply_ufunc(
        lambda x: model.neural_net.forward(torch.from_numpy(x).float())
        .detach()
        .numpy(),
        plot_grid,
        vectorize=True,
        input_core_dims=[["idx"]],
    )

    log.debug("   Evaluating the solution on the grid ...")
    u_exact = xr.apply_ufunc(
        this.EXAMPLES[model_cfg["PDE"]["function"]]["u"],
        plot_grid,
        input_core_dims=[["idx"]],
        vectorize=True,
        keep_attrs=True,
    )

    dset_u_exact = h5group.create_dataset(
        "u_exact",
        list(u_exact.sizes.values()),
        maxshape=list(u_exact.sizes.values()),
        chunks=True,
        compression=3,
    )
    dset_u_exact.attrs["dim_names"] = list(u_exact.sizes)

    # Set attributes
    for idx in list(u_exact.sizes):
        dset_u_exact.attrs["coords_mode__" + str(idx)] = "values"
        dset_u_exact.attrs["coords__" + str(idx)] = u_exact.coords[idx].data

    # Write data
    dset_u_exact[
        :,
    ] = u_exact

    dset_predictions = h5group.create_dataset(
        "predictions",
        list(predictions.sizes.values()),
        maxshape=list(predictions.sizes.values()),
        chunks=True,
        compression=3,
    )
    dset_predictions.attrs["dim_names"] = list(predictions.sizes)

    # Set the attributes
    for idx in list(predictions.sizes):
        dset_predictions.attrs["coords_mode__" + str(idx)] = "values"
        dset_predictions.attrs["coords__" + str(idx)] = plot_grid.coords[idx].data

    dset_predictions[
        :,
    ] = predictions

    log.info("   Done. Wrapping up ...")
    h5file.close()

    log.success("   All done.")
