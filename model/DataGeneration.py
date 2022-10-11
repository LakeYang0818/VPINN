import logging
import sys
from os.path import dirname as up
from typing import Sequence, Union

import torch
from dantro._import_tools import import_module_from_path

log = logging.getLogger(__name__)

sys.path.append(up(up(__file__)))
sys.path.append(up(up(__file__)))

base = import_module_from_path(mod_path=up(up(__file__)), mod_str="include")
this = import_module_from_path(mod_path=up(__file__), mod_str="model")

# ----------------------------------------------------------------------------------------------------------------------
# -- Load or generate grid and training data ---------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# TODO Allow loading data from file


def get_data(space_dict, test_func_dict, pde_cfg, *, var_form: int, h5group) -> dict:
    """Returns the grid and test function data, either by loading it from a file or generating it.

    If generated, data is to written to the output folder
    :param space_dict: the dictionary containing the space configuration
    :param test_func_dict: the dictionary containing the test function configuration
    :param var_form: the variational form to use
    :param h5group: the h5group to write data to
    :return: data: a dictionary containing the grid and test function data
    """

    dim = space_dict["dimension"]

    # Get type and number of test functions in each dimension
    test_func_type: str = test_func_dict["type"]
    test_func_dim: Union[int, Sequence[int]] = (
        test_func_dict["N_test_functions"]["x"]
        if dim == 1
        else [
            test_func_dict["N_test_functions"]["x"],
            test_func_dict["N_test_functions"]["y"],
        ]
    )
    n_test_funcs: int = (
        test_func_dim if dim == 1 else test_func_dim[0] * test_func_dim[1]
    )

    # Construct the grid
    log.info("   Constructing grid ...")
    grid: base.Grid = base.construct_grid_from_cfg(
        space_dict,
        as_tensor=True,
        requires_grad=False,
        requires_normals=(var_form >= 2),
    )

    # Evaluate the test functions and any required derivatives on the grid interior
    log.info("   Evaluating test functions on the grid interior ... ")
    test_func_vals: base.DataSet = base.testfunc_grid_evaluation(
        grid, test_func_dim, d=0, where="interior", which=test_func_type
    )

    d1test_func_vals: base.DataSet = (
        base.testfunc_grid_evaluation(
            grid, test_func_dim, d=1, where="interior", which=test_func_type
        )
        if var_form >= 1
        else None
    )

    d2test_func_vals: base.DataSet = (
        base.testfunc_grid_evaluation(
            grid, test_func_dim, d=2, where="interior", which=test_func_type
        )
        if var_form >= 2
        else None
    )

    # Evaluate the test functions on the grid boundary
    d1test_func_vals_bd: base.DataSet = (
        base.testfunc_grid_evaluation(
            grid, test_func_dim, d=1, where="boundary", which=test_func_type
        )
        if var_form >= 2
        else None
    )

    # Integrate the external function over the grid against all the test functions.
    # This will be used to calculate the variational loss; this step is costly and only needs to be done once.
    log.info("   Integrating test functions ...")
    f_integrated: base.DataSet = base.DataSet(
        coords=test_func_vals.coords,
        data=[
            base.integrate(
                this.f(grid.interior, func=pde_cfg["function"]),
                test_func_vals.data[i],
                domain_volume=grid.volume,
            )
            for i in range(n_test_funcs)
        ],
        as_tensor=True,
        requires_grad=False,
    )

    # TODO This can be done automatically with xarray attributes
    if dim == 2:
        tf_data = torch.reshape(
            test_func_vals.data,
            (test_func_dim[1], test_func_dim[0], len(grid.y) - 2, len(grid.x) - 2, 1),
        )
        tf_dims = ["n_y", "n_x", "y", "x", "idx"]
    else:
        tf_data = test_func_vals.data
        tf_dims = ["n_x", "x", "idx"]

    # Store the test functions
    dset_test_func_vals = h5group.create_dataset(
        "test_function_values",
        tf_data.shape,
        maxshape=tf_data.shape,
        chunks=True,
        compression=3,
    )
    dset_test_func_vals.attrs["dim_names"] = tf_dims

    dset_test_func_vals.attrs["coords_mode__n_x"] = "trivial"
    dset_test_func_vals.attrs["coords_moe__idx"] = "trivial"
    dset_test_func_vals.attrs["coords_mode__x"] = "values"
    dset_test_func_vals.attrs["coords__x"] = torch.flatten(grid.x)[1:-1]

    if dim == 2:
        dset_test_func_vals.attrs["coords_mode__n_y"] = "trivial"
        dset_test_func_vals.attrs["coords_mode__y"] = "values"
        dset_test_func_vals.attrs["coords__y"] = torch.flatten(grid.y)[1:-1]

    dset_test_func_vals[
        :,
    ] = tf_data

    return dict(
        grid=grid,
        test_func_vals=test_func_vals,
        d1test_func_vals=d1test_func_vals,
        d2test_func_vals=d2test_func_vals,
        d1test_func_vals_bd=d1test_func_vals_bd,
        f_integrated=f_integrated,
    )
