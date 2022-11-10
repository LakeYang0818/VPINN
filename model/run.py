#!/usr/bin/env python3
import sys
import time
from os.path import dirname as up
from typing import Union

import coloredlogs
import h5py as h5
import numpy as np
import ruamel.yaml as yaml
import torch
import utopya_backend
import xarray as xr
from dantro import logging
from dantro._import_tools import import_module_from_path
from paramspace.tools import recursive_update

sys.path.append(up(__file__))
sys.path.append(up(up(__file__)))

base = import_module_from_path(mod_path=up(up(__file__)), mod_str="include")
this = import_module_from_path(mod_path=up(__file__), mod_str="model")

log = logging.getLogger(__name__)
coloredlogs.install(fmt="%(levelname)s %(message)s", level="DEBUG", logger=log)


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
        write_predictions_every: int = -1,
        grid: xr.DataArray,
        training_data: xr.Dataset = None,
        f_integrated: xr.DataArray,
        test_function_values: xr.DataArray,
        d1_test_function_values: xr.DataArray = None,
        d2_test_function_values: xr.DataArray = None,
        d1_test_function_values_boundary: xr.DataArray = None,
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

            """Unpacks a DataArray of test function values and returns a stacked torch.Tensor. Shape of the
            output is given by: [test function multi-index, coordinate multi-index, 1]"""

            if test_funcs is None:
                return None

            return (
                torch.reshape(
                    torch.from_numpy(
                        test_funcs.isel(
                            {
                                var: slice(1, -1)
                                for var in test_funcs.attrs["space_dimensions"]
                            }
                        ).data
                    ),
                    (len(test_funcs.coords["tf_idx"]), -1, 1),
                )
                .float()
                .to(device)
            )

        def _dtf_to_tensor(
            test_funcs: xr.DataArray = None,
        ) -> Union[None, torch.Tensor]:

            """Unpacks a DataArray of test function derivatives and returns a stacked torch.Tensor"""

            if test_funcs is None:
                return None

            return (
                torch.reshape(
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
                )
                .float()
                .to(device)
            )

        def _get_normals(ds: xr.Dataset = None) -> Union[None, torch.Tensor]:

            """Unpacks the Dataset containing the grid normals and returns a stacked torch.Tensor"""

            if ds is None:
                return None

            res = []
            for dim in ds.attrs["space_dimensions"]:
                res.append(
                    torch.from_numpy(
                        training_data.sel(
                            variable=["normals_" + str(dim)], drop=True
                        ).boundary_data.to_numpy()
                    ).float()
                )

            return torch.reshape(
                torch.stack(res, dim=1), (-1, ds.attrs["grid_dimension"])
            ).to(device)

        self._name = name
        self._time = 0
        self._h5group = h5group
        self._rng = rng
        self.device = device

        self.neural_net = neural_net.to(device)
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
        self._write_predictions_every = write_predictions_every

        # The grid interior
        self.grid_interior: torch.Tensor = (
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
                ).boundary_data.to_numpy()
            )
            .float()
            .to(device)
        )

        # The grid normals
        self.grid_normals: torch.Tensor = _get_normals(training_data).to(device)

        # The density of the grid
        self.domain_density = grid.attrs["grid_density"]

        # Training data (boundary conditions)
        self.training_data: torch.Tensor = (
            torch.from_numpy(
                training_data.sel(variable=["u"], drop=True).boundary_data.to_numpy()
            )
            .float()
            .to(device)
        )

        # Value of the external function integrated against all the test functions
        self.f_integrated: torch.Tensor = torch.reshape(
            torch.from_numpy(f_integrated.data).float(), (-1, 1)
        ).to(device)

        # Test function values on the grid interior, indexed by their (multi-)index and grid coordinate
        self.test_func_values: torch.Tensor = _tf_to_tensor(test_function_values)

        self.d1test_func_values: Union[None, torch.Tensor] = _dtf_to_tensor(
            d1_test_function_values
        )

        self.d2test_func_values: Union[None, xr.DataArray] = _dtf_to_tensor(
            d2_test_function_values
        )

        self.d1test_func_values_boundary: torch.Tensor = (
            torch.from_numpy(d1_test_function_values_boundary.to_array().squeeze().data)
            .float()
            .to(device)
        )

        self.weights = (
            torch.reshape(
                torch.stack(
                    [
                        weight_function(np.array(idx))
                        for idx in test_function_values.coords["tf_idx"].data
                    ]
                ),
                (-1, 1),
            )
            .float()
            .to(device)
        )

        # Store current predictions, if specified
        if self._write_predictions_every > 0:

            # The grid interior
            self.grid: torch.Tensor = (
                torch.reshape(
                    torch.from_numpy(grid.to_numpy()).float(),
                    (-1, grid.attrs["grid_dimension"]),
                )
                .requires_grad_(True)
                .to(device)
            )

            self._dset_current_predictions = self._h5group.create_dataset(
                "predictions_over_time",
                (0,) + self.grid.shape,
                maxshape=(None,) + self.grid.shape,
                chunks=True,
                compression=3,
            )
            self._dset_current_predictions.attrs["dim_names"] = ["time"] + list(
                grid.sizes
            )
            self._dset_current_predictions.attrs["coords_mode__time"] = "start_and_step"
            self._dset_current_predictions.attrs["coords__time"] = [
                write_start,
                write_predictions_every,
            ]

            # Set attributes
            for idx in list(grid.sizes):
                self._dset_current_predictions.attrs[
                    "coords_mode__" + str(idx)
                ] = "values"
                self._dset_current_predictions.attrs[
                    "coords__" + str(idx)
                ] = grid.coords[idx].data

    def epoch(
        self, *, boundary_loss_weight: float = 1.0, variational_loss_weight: float = 1.0
    ):

        """Trains the model for a single epoch"""

        start_time = time.time()

        # Reset the neural net optimizer
        self.neural_net.optimizer.zero_grad()

        # Calculate the boundary loss
        boundary_loss = torch.sum(
            torch.square(
                self.neural_net.forward(self.grid_boundary) - self.training_data
            ),
            dim=0,
        )

        variational_loss = self.neural_net.variational_loss(
            self.device,
            self.grid_interior,
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
        self.current_loss = loss.clone().detach().cpu().numpy().item()
        self.current_boundary_loss = boundary_loss.clone().detach().cpu().numpy().item()
        self.current_variational_loss = (
            variational_loss.clone().detach().cpu().numpy().item()
        )

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

        if self._write_predictions_every > 0:
            if self._time >= self._write_start and (
                self._time % self._write_predictions_every == 0
            ):

                self._dset_current_predictions.resize(
                    self._dset_current_predictions.shape[0] + 1, axis=0
                )
                self._dset_current_predictions[-1, :] = (
                    self.neural_net.forward(self.grid).detach().numpy()
                )


if __name__ == "__main__":

    cfg_file_path = sys.argv[1]
    log.note("   Preparing model run ...")
    log.note(f"   Loading config file:\n        {cfg_file_path}")
    with open(cfg_file_path) as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.Loader)
    model_name = cfg.get("root_model_name", "VPINN")
    log.note(f"   Model name:  {model_name}")
    model_cfg = cfg[model_name]
    logging.getLogger().setLevel(utopya_backend.get_level(cfg["log_levels"]["model"]))

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

    # Get the data: grid, test function data, and training data. This is loaded from a file,
    # if provided, else synthetically generated
    data: dict = this.get_grid_tf_data(
        model_cfg.get("load_data", {}),
        model_cfg["space"],
        model_cfg["test_functions"],
        h5file=h5file,
    )

    # If a data generation run was performed, return
    if cfg.get("generation_run", False):
        log.success("   Grid and test function data generated.")
        h5file.close()
        sys.exit(0)

    # Get the training data
    data.update(
        this.get_training_data(
            func=this.EXAMPLES[model_cfg["PDE"]["function"]]["u"],
            grid=data["grid"],
            boundary=data["grid_boundary"],
            boundary_isel=model_cfg["Training"].get("boundary", None),
        )
    )
    # Get the external forcing data
    data.update(
        this.get_forcing_data(
            func=this.EXAMPLES[model_cfg["PDE"]["function"]]["f"],
            grid=data["grid"],
            test_function_values=data["test_function_values"],
        )
    )

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
        input_size=data["grid"].attrs["grid_dimension"],
        output_size=1,
        eq_type=eq_type,
        var_form=var_form,
        pde_constants=PDE_constants,
        **model_cfg["NeuralNet"],
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
        write_predictions_every=cfg["write_predictions_every"],
        weight_function=this.WEIGHT_FUNCTIONS[
            model_cfg["test_functions"]["weight_function"].lower()
        ],
        **data,
    )

    # Train the model
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

    # Generate a prediction using the trained network. The prediction is evaluated on a separate grid,
    # which can be finer than the training grid, if specified. Predictions are generated on the CPU.
    log.debug("   Generating plot grid ...")
    plot_grid = base.construct_grid(
        recursive_update(
            model_cfg.get("space", {}), model_cfg.get("predictions_grid", {})
        )
    )

    net = net.to("cpu")

    log.debug("   Evaluating the prediction on the plot grid ...")
    predictions = xr.apply_ufunc(
        lambda x: net.forward(torch.tensor(x).float()).detach().numpy(),
        plot_grid,
        vectorize=True,
        input_core_dims=[["idx"]],
    )

    # Evaluate the solution on the grid
    log.debug("   Evaluating the solution on the plot grid ...")
    u_exact = xr.apply_ufunc(
        this.EXAMPLES[model_cfg["PDE"]["function"]]["u"],
        plot_grid,
        input_core_dims=[["idx"]],
        vectorize=True,
        keep_attrs=True,
    )

    # Save training and prediction data
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
        "prediction_final",
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

    # External function evaluated on the grid
    f_evaluated = data["f_evaluated"]
    dset_f_evaluated = h5group.create_dataset(
        "f_evaluated",
        list(f_evaluated.sizes.values()),
        maxshape=list(f_evaluated.sizes.values()),
        chunks=True,
        compression=3,
    )
    dset_f_evaluated.attrs["dim_names"] = [str(_) for _ in list(f_evaluated.sizes)]

    # Set the attributes
    for idx in list(f_evaluated.sizes):
        dset_f_evaluated.attrs["coords_mode__" + str(idx)] = "values"
        dset_f_evaluated.attrs["coords__" + str(idx)] = data["grid"].coords[idx].data
    dset_f_evaluated.attrs.update(f_evaluated.attrs)

    # Write the data
    dset_f_evaluated[
        :,
    ] = f_evaluated

    # Integral of the forcing against the test functions.
    # This dataset is indexed by the test function indices
    f_integrated = data["f_integrated"].unstack()
    dset_f_integrated = h5group.create_dataset(
        "f_integrated",
        list(f_integrated.sizes.values()),
        maxshape=list(f_integrated.sizes.values()),
        chunks=True,
        compression=3,
    )
    dset_f_integrated.attrs["dim_names"] = [str(_) for _ in list(f_integrated.sizes)]

    # Set attributes
    for idx in list(f_integrated.sizes):
        dset_f_integrated.attrs["coords_mode__" + str(idx)] = "values"
        dset_f_integrated.attrs["coords__" + str(idx)] = f_integrated.coords[idx].data
    dset_f_integrated.attrs.update(f_integrated.attrs)

    # Write data
    dset_f_integrated[
        :,
    ] = f_integrated

    # Training data: values of the test function on the boundary
    training_data = data["training_data"]
    dset_training_data = h5group.create_dataset(
        "training_data",
        list(training_data.sizes.values()),
        maxshape=list(training_data.sizes.values()),
        chunks=True,
        compression=3,
    )
    dset_training_data.attrs["dim_names"] = [str(_) for _ in list(training_data.sizes)]
    dset_training_data.attrs.update(training_data.attrs)

    # Set attributes
    dset_training_data.attrs["coords_mode__idx"] = "trivial"
    dset_training_data.attrs["coords_mode__variable"] = "values"
    dset_training_data.attrs["coords__variable"] = [
        str(_) for _ in training_data.coords["variable"].data
    ]

    # Write data
    dset_training_data[
        :,
    ] = training_data.to_array()

    log.info("   Done. Wrapping up ...")
    h5file.close()

    log.success("   All done.")
