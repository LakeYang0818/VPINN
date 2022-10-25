import copy
import logging
import sys
from os.path import dirname as up
from typing import Union

import h5py as h5
import paramspace
import xarray as xr
from dantro._import_tools import import_module_from_path

log = logging.getLogger(__name__)

sys.path.append(up(up(__file__)))

base = import_module_from_path(mod_path=up(up(__file__)), mod_str="include")


# ----------------------------------------------------------------------------------------------------------------------
# -- Load or generate grid and training data ---------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def get_data(
    load_from_file: str,
    space_dict: dict,
    test_func_dict: dict,
    *,
    solution: callable,
    forcing: callable,
    var_form: int,
    boundary_isel: Union[str, tuple] = None,
    h5file: h5.File,
) -> dict:

    """Returns the grid and test function data, either by loading it from a file or generating it.
    If generated, data is to written to the output folder

    :param load_from_file: the path to the data file. If none, data is automatically generated
    :param space_dict: the dictionary containing the space configuration
    :param test_func_dict: the dictionary containing the test function configuration
    :param solution: the explicit solution (to be evaluated on the grid boundary)
    :param forcing: the external function
    :param var_form: the variational form to use
    :param boundary_isel: (optional) section of the boundary to use for training. Can either be a string ('lower',
        'upper', 'left', 'right') or a range.
    :param h5file: the h5file to write data to
    :return: data: a dictionary containing the grid and test function data
    """

    data = {}

    if load_from_file is not None:

        log.info("   Loading data ...")
        data = {}

        with h5.File(load_from_file, "r") as f:
            data["grid"] = f["grid"]
            data["test_func_values"] = f["test_function_values"]

            if var_form >= 1:
                data["d1test_func_values"] = f["d1_test_function_values"]

            if var_form >= 2:
                data["d2test_func_values"] = f["d2_test_function_values"]
                data["d1test_func_values_bd"] = f["d1_test_function_values_boundary"]

        log.info("   All data loaded")

    else:

        # --------------------------------------------------------------------------------------------------------------
        # --- Generate data --------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        if len(space_dict) != len(test_func_dict["num_functions"]):
            raise ValueError(
                f"Space and test function dimensions do not match! "
                f"Got {len(space_dict)} and {len(test_func_dict['num_functions'])}."
            )

        log.info("   Generating data ...")

        log.debug("   Constructing the grid ... ")
        grid: xr.DataArray = base.construct_grid(space_dict)
        boundary: xr.Dataset = base.get_boundary(grid)
        data["grid"] = grid
        data["boundary"] = boundary
        training_boundary = (
            boundary
            if boundary_isel is None
            else base.get_boundary_isel(boundary, boundary_isel, grid)
        )
        log.note("   Constructed the grid.")

        # The test functions are only defined on [-1, 1]
        # TODO Generating two grids can be expensive!
        tf_space_dict = paramspace.tools.recursive_replace(
            copy.deepcopy(space_dict),
            select_func=lambda d: "extent" in d,
            replace_func=lambda d: d.update(dict(extent=[-1, 1])) or d,
        )

        tf_grid: xr.DataArray = base.construct_grid(tf_space_dict)
        tf_boundary: xr.Dataset = base.get_boundary(tf_grid)

        log.debug("   Evaluating test functions on grid ...")
        # The test functions are defined on
        # TODO the test functions only need to be caluclated on the coordinates
        test_function_indices = base.construct_grid(
            test_func_dict["num_functions"], lower=1, dtype=int
        )
        test_function_values = base.tf_grid_evaluation(
            tf_grid, test_function_indices, type=test_func_dict["type"], d=0
        )

        log.note("   Evaluated the test functions.")
        data["test_func_values"] = test_function_values.stack(
            tf_idx=test_function_values.attrs["test_function_dims"]
        )

        log.debug("   Evaluating test function derivatives on grid ... ")
        d1test_func_values = base.tf_grid_evaluation(
            tf_grid, test_function_indices, type=test_func_dict["type"], d=1
        )

        data["d1test_func_values"] = d1test_func_values.stack(
            tf_idx=test_function_values.attrs["test_function_dims"]
        )

        d1test_func_values_boundary = base.tf_simple_evaluation(
            tf_boundary.sel(variable=tf_grid.attrs["space_dimensions"]),
            test_function_indices,
            type=test_func_dict["type"],
            d=1,
            core_dim="variable",
        )
        data["d1test_func_values_boundary"] = d1test_func_values_boundary.stack(
            tf_idx=test_function_values.attrs["test_function_dims"]
        ).transpose("tf_idx", ...)

        log.debug("   Evaluating test function second derivatives on grid ... ")
        d2test_func_values = base.tf_grid_evaluation(
            tf_grid, test_function_indices, type=test_func_dict["type"], d=2
        )
        data["d2test_func_values"] = d2test_func_values.stack(
            tf_idx=test_function_values.attrs["test_function_dims"]
        )

        log.debug("   Evaluating the external function on the grid ...")
        f_evaluated: xr.DataArray = xr.apply_ufunc(
            forcing, grid, input_core_dims=[["idx"]], vectorize=True
        )
        data["f_evaluated"] = f_evaluated

        log.debug("   Integrating the function over the grid ...")
        f_integrated = base.integrate_xr(f_evaluated, test_function_values)
        data["f_integrated"] = f_integrated.stack(
            tf_idx=test_function_values.attrs["test_function_dims"]
        )

        log.debug("   Evaluating the solution on the boundary ...")
        u_boundary: xr.Dataset = xr.concat(
            [
                training_boundary,
                xr.apply_ufunc(
                    solution,
                    training_boundary.sel(variable=grid.attrs["space_dimensions"]),
                    input_core_dims=[["variable"]],
                    vectorize=True,
                ).assign_coords(variable=("variable", ["u"])),
            ],
            dim="variable",
        )
        data["training_data"] = u_boundary

        # --------------------------------------------------------------------------------------------------------------
        # --- Set up chunked dataset to store the state data in --------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        data_group = h5file.create_group("data")

        # --------------------------------------------------------------------------------------------------------------
        # --- Grid -----------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        dset_grid = data_group.create_dataset(
            "grid",
            list(grid.sizes.values()),
            maxshape=list(grid.sizes.values()),
            chunks=True,
            compression=3,
        )
        dset_grid.attrs["dim_names"] = list(grid.sizes)

        # Set attributes
        for idx in list(grid.sizes):
            dset_grid.attrs["coords_mode__" + str(idx)] = "values"
            dset_grid.attrs["coords__" + str(idx)] = grid.coords[idx].data

        dset_grid[
            :,
        ] = grid

        # Training data: values of the test function on the boundary
        dset_boundary = data_group.create_dataset(
            "grid_boundary",
            [
                len(boundary),
                list(boundary.sizes.values())[0],
                list(boundary.sizes.values())[1],
            ],
            maxshape=[
                len(boundary),
                list(boundary.sizes.values())[0],
                list(boundary.sizes.values())[1],
            ],
            chunks=True,
            compression=3,
        )
        dset_boundary.attrs["dim_names"] = ["dim_name__0", "idx", "variable"]

        # Set attributes
        dset_boundary.attrs["coords_mode__idx"] = "trivial"
        dset_boundary.attrs["coords_mode__variable"] = "values"
        dset_boundary.attrs["coords__variable"] = [
            str(_) for _ in boundary.coords["variable"].data
        ]

        # Write data
        dset_boundary[
            :,
        ] = boundary.to_array()

        # --------------------------------------------------------------------------------------------------------------
        # --- Test function values -------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # Store the test function values
        dset_test_func_vals = data_group.create_dataset(
            "test_function_values",
            test_function_values.shape,
            maxshape=test_function_values.shape,
            chunks=True,
            compression=3,
        )
        dset_test_func_vals.attrs["dim_names"] = list(test_function_values.sizes)

        # Store the first derivatives of the test function values
        dset_d1_test_func_vals = data_group.create_dataset(
            "d1_test_function_values",
            d1test_func_values.shape,
            maxshape=d1test_func_values.shape,
            chunks=True,
            compression=3,
        )
        dset_d1_test_func_vals.attrs["dim_names"] = list(d1test_func_values.sizes)

        # Store the second derivatives of the test function values
        dset_d2_test_func_vals = data_group.create_dataset(
            "d2_test_function_values",
            d2test_func_values.shape,
            maxshape=d2test_func_values.shape,
            chunks=True,
            compression=3,
        )
        dset_d2_test_func_vals.attrs["dim_names"] = list(d2test_func_values.sizes)

        # Set attributes
        for idx in list(test_function_values.sizes):
            dset_test_func_vals.attrs["coords_mode__" + str(idx)] = "values"
            dset_test_func_vals.attrs[
                "coords__" + str(idx)
            ] = test_function_values.coords[idx].data

            dset_d1_test_func_vals.attrs["coords_mode__" + str(idx)] = "values"
            dset_d1_test_func_vals.attrs[
                "coords__" + str(idx)
            ] = d1test_func_values.coords[idx].data

            dset_d2_test_func_vals.attrs["coords_mode__" + str(idx)] = "values"
            dset_d2_test_func_vals.attrs[
                "coords__" + str(idx)
            ] = d2test_func_values.coords[idx].data

        # Write the data
        dset_test_func_vals[
            :,
        ] = test_function_values
        dset_d1_test_func_vals[
            :,
        ] = d1test_func_values
        dset_d2_test_func_vals[
            :,
        ] = d2test_func_values

        # --------------------------------------------------------------------------------------------------------------
        # --- External forcing -----------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # Store the forcing evaluated on the grid
        dset_f_evaluated = data_group.create_dataset(
            "f_evaluated",
            list(f_evaluated.sizes.values()),
            maxshape=list(f_evaluated.sizes.values()),
            chunks=True,
            compression=3,
        )
        dset_f_evaluated.attrs["dim_names"] = list(f_evaluated.sizes)

        # Set the attributes
        for idx in list(f_evaluated.sizes):
            dset_f_evaluated.attrs["coords_mode__" + str(idx)] = "values"
            dset_f_evaluated.attrs["coords__" + str(idx)] = grid.coords[idx].data

        dset_f_evaluated[
            :,
        ] = f_evaluated

        # Store the integral of the forcing against the test functions. This dataset is indexed by the
        # test function indices
        dset_f_integrated = data_group.create_dataset(
            "f_integrated",
            list(f_integrated.sizes.values()),
            maxshape=list(f_integrated.sizes.values()),
            chunks=True,
            compression=3,
        )
        dset_f_integrated.attrs["dim_names"] = list(f_integrated.sizes)

        for idx in list(f_integrated.sizes):
            dset_f_integrated.attrs["coords_mode__" + str(idx)] = "values"
            dset_f_integrated.attrs["coords__" + str(idx)] = f_integrated.coords[
                idx
            ].data
        dset_f_integrated[
            :,
        ] = f_integrated

        # --------------------------------------------------------------------------------------------------------------
        # --- Exact solution -------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # Training data: values of the test function on the boundary
        dset_u_exact_bd = data_group.create_dataset(
            "u_exact_boundary",
            [
                len(u_boundary),
                list(u_boundary.sizes.values())[0],
                list(u_boundary.sizes.values())[1],
            ],
            maxshape=[
                len(u_boundary),
                list(u_boundary.sizes.values())[0],
                list(u_boundary.sizes.values())[1],
            ],
            chunks=True,
            compression=3,
        )
        dset_u_exact_bd.attrs["dim_names"] = ["dim_name__0", "idx", "variable"]

        # Set attributes
        dset_u_exact_bd.attrs["coords_mode__idx"] = "trivial"
        dset_u_exact_bd.attrs["coords_mode__variable"] = "values"
        dset_u_exact_bd.attrs["coords__variable"] = [
            str(_) for _ in u_boundary.coords["variable"].data
        ]

        # Write data
        dset_u_exact_bd[
            :,
        ] = u_boundary.to_array()

        log.info("   All data generated and saved.")

    return data
