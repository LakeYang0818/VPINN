#!/usr/bin/env python3
import sys
import time
from os.path import dirname as up
from typing import Sequence, Union

import coloredlogs
import h5py as h5
import numpy as np
import ruamel.yaml as yaml
import torch
from dantro import logging
from dantro._import_tools import import_module_from_path
from paramspace.tools import recursive_update

sys.path.append(up(__file__))
sys.path.append(up(up(__file__)))

base = import_module_from_path(mod_path=up(up(__file__)), mod_str="include")
this = import_module_from_path(mod_path=up(__file__), mod_str="model")

log = logging.getLogger(__name__)
coloredlogs.install(fmt="%(levelname)s %(message)s", level="INFO", logger=log)

# ......................................................................................................................

if __name__ == "__main__":

    cfg_file_path = sys.argv[1]
    log.note("   Preparing model run ...")
    log.note(f"   Loading config file:\n        {cfg_file_path}")
    with open(cfg_file_path, "r") as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.Loader)
    model_name = cfg.get("root_model_name", "VPINN")
    log.note(f"   Model name:  {model_name}")
    model_cfg = cfg[model_name]
    base.validate_cfg(model_cfg)

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

    # Set up chunked dataset to store the loss data in
    dset_loss = h5group.create_dataset(
        "loss",
        (0, 3, 1),
        maxshape=(None, 3, 1),
        chunks=True,
        compression=3,
    )
    dset_loss.attrs["dim_names"] = ["time", "loss type", "dim_name__3"]
    dset_loss.attrs["coords_mode__time"] = "start_and_step"
    dset_loss.attrs["coords__time"] = [0, 1]
    dset_loss.attrs["coords_mode__loss type"] = "values"
    dset_loss.attrs["coords__loss type"] = [
        "total loss",
        "boundary loss",
        "variational loss",
    ]

    # Get space parameters from the config
    dim: int = model_cfg["space"]["dimension"]
    eq_type: str = model_cfg["PDE"]["type"]
    var_form: int = model_cfg["variational_form"]

    # Get PDE constants from the config
    PDE_constants: dict = {
        "Burger": model_cfg["PDE"]["Burger"]["nu"],
        "Helmholtz": model_cfg["PDE"]["Helmholtz"]["k"],
        "PorousMedium": model_cfg["PDE"]["PorousMedium"]["m"],
    }

    # Get all the necessary data
    data: dict = this.get_data(
        model_cfg["space"],
        model_cfg["test_functions"],
        model_cfg["PDE"],
        var_form=var_form,
        h5group=h5group,
    )
    grid = data["grid"]

    # The weight function for the test functions.
    weight_function = this.WEIGHT_FUNCTIONS[
        model_cfg["test_functions"]["weight_function"]
    ]

    # Instantiate the model class
    neural_net: base.NeuralNet = base.NeuralNet(
        eq_type,
        var_form,
        pde_constants=PDE_constants,
        input_size=model_cfg["space"]["dimension"],
        output_size=1,
        **model_cfg["NeuralNet"],
    ).to(device)

    # Turn on tracking for the grid interior, on which the variational loss is calculated
    grid.interior.requires_grad = True

    # Train the model
    num_epochs = cfg["num_epochs"]
    log.info(f"   Now commencing training for {num_epochs} epochs ...")
    b_weight, v_weight = (
        model_cfg["Training"]["boundary_loss_weight"],
        model_cfg["Training"]["variational_loss_weight"],
    )
    start_time = time.time()
    for it in range(num_epochs):

        neural_net.optimizer.zero_grad()

        # Calculate the loss
        loss_b = neural_net.boundary_loss(data["training_data"])
        loss_v = neural_net.variational_loss(
            grid,
            data["f_integrated"],
            data["test_func_vals"],
            data["d1test_func_vals"],
            data["d2test_func_vals"],
            data["d1test_func_vals_bd"],
            weight_function,
        )
        loss = b_weight * loss_b + v_weight * loss_v
        loss.backward()

        # Adjust the model parameters
        neural_net.optimizer.step()

        # Track loss values
        dset_loss.resize(dset_loss.shape[0] + 1, axis=0)
        dset_loss[-1, :] = [
            [loss.item()],
            [loss_b.item()],
            [loss_v.item()],
        ]
        if it % 100 == 0:
            log.progress(
                f"   Epoch {it}/{num_epochs}: total loss: {loss.item()}, boundary loss: {loss_b.item()}, "
                f"variational loss: {loss_v.item()}"
            )

        del loss

    log.info(
        f"   Training completed in {np.around(time.time() - start_time, 3)} seconds."
    )

    # Get the plot grid, which can be finer than the training grid, if specified
    plot_grid: base.Grid = base.construct_grid_from_cfg(
        recursive_update(model_cfg["space"], model_cfg.get("predictions_grid", {})),
        requires_grad=False,
    )

    # Get the model predictions on the plotting grid. Turn off tracking for the prediction data.
    predictions = torch.reshape(
        neural_net.forward(plot_grid.data).detach(), (len(plot_grid.x), -1)
    )

    # ------------------------------------------------------------------------------------------------------------------
    # --- Data writing -------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    space_dims = ["x"] if dim == 1 else ["y", "x"]
    dset_predictions = h5group.create_dataset(
        "predictions",
        predictions.shape,
        maxshape=predictions.shape,
        chunks=True,
        compression=3,
    )
    dset_predictions.attrs["dim_names"] = (
        space_dims + ["idx"] if dim == 1 else space_dims
    )

    # Set the attributes
    for idx in plot_grid.space_dims:
        dset_predictions.attrs["coords_mode__" + str(idx)] = "values"
        dset_predictions.attrs["coords__" + str(idx)] = (
            plot_grid.x.flatten() if idx == "x" else plot_grid.y.flatten()
        )

    dset_predictions[
        :,
    ] = predictions

    # Store the exact solution
    u_exact = this.u(plot_grid.data, func=model_cfg["PDE"]["function"])
    if dim == 2:
        u_exact = torch.reshape(u_exact, (-1, len(plot_grid.x)))

    dset_u_exact = h5group.create_dataset(
        "u_exact",
        u_exact.shape,
        maxshape=u_exact.shape,
        chunks=True,
        compression=3,
    )

    dset_u_exact.attrs["dim_names"] = space_dims + ["idx"] if dim == 1 else space_dims
    dset_u_exact.attrs["coords_mode__x"] = "values"
    dset_u_exact.attrs["coords__x"] = torch.flatten(plot_grid.x)

    if dim == 2:
        dset_u_exact.attrs["coords_mode__y"] = "values"
        dset_u_exact.attrs["coords__y"] = torch.flatten(plot_grid.y)

    dset_u_exact[
        :,
    ] = u_exact

    log.info("   Done. Wrapping up ...")
    h5file.close()

    log.success("   All done.")
