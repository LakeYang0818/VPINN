import copy
import logging
import sys
from os.path import dirname as up
from typing import Union

import h5py as h5
import paramspace
import xarray as xr
from dantro._import_tools import import_module_from_path

import utopya.eval.datamanager

log = logging.getLogger(__name__)

sys.path.append(up(up(__file__)))

base = import_module_from_path(mod_path=up(up(__file__)), mod_str="include")


# ----------------------------------------------------------------------------------------------------------------------
# -- Load or generate grid and training data ---------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def get_data(
    load_cfg: dict,
    space_dict: dict,
    test_func_dict: dict,
    *,
    solution: callable,
    forcing: callable,
    boundary_isel: Union[str, tuple] = None,
    h5file: h5.File,
) -> dict:

    """Returns the grid and test function data, either by loading it from a file or generating it.
    If generated, data is to written to the output folder

    :param load_cfg: load configuration, containing the path to the data file. If the path is None, data is
         automatically generated. If the ``copy_data`` entry is true, data will be copied and written to the
         output directory; however, this is false by default to save disk space.
         The configuration also contains further kwargs, passed to the loader.
    :param space_dict: the dictionary containing the space configuration
    :param test_func_dict: the dictionary containing the test function configuration
    :param solution: the explicit solution (to be evaluated on the grid boundary)
    :param forcing: the external function
    :param var_form: the variational form to use
    :param boundary_isel: (optional) section of the boundary to use for training. Can either be a string ('lower',
        'upper', 'left', 'right') or a range.
    :param h5file: the h5file to write data to. A new group is added to this file.

    :return: data: a dictionary containing the grid and test function data
    """

    # TODO: allow selection of which data to load
    # TODO: allow passing Sequences of boundary sections, e.g. ['lower', 'upper']
    # TODO: Do not generate separate h5Group?
    # TODO: re-writing loaded data not possible

    # Collect datasets in a dictionary, passed to the model
    data = {}

    # The directory from which to load data
    data_dir = load_cfg.pop("data_dir", None)

    if data_dir is not None:

        # --------------------------------------------------------------------------------------------------------------
        # --- Load data ------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        log.info("   Loading data ...")

        # If data is loaded, determine whether to copy that data to the new output directory.
        # This is false by default to save disk storage
        copy_data = load_cfg.pop("copy_data", False)

        dm = utopya.eval.datamanager.DataManager(data_dir=data_dir, out_dir=False)

        dm.load(
            "VPINN_data",
            loader=load_cfg.pop("loader", "hdf5"),
            glob_str=load_cfg.pop("glob_str", "*.h5"),
            print_tree=load_cfg.pop("print_tree", False),
            **load_cfg,
        )

        for key in list(dm["VPINN_data"]["data"]["data"].keys()):

            # Get the dataset and convert to xr.DataArray
            ds = dm["VPINN_data"]["data"]["data"][key]
            data[key] = ds.data

            # These entries should be xr.Datasets
            if key in [
                "training_data",
                "grid_boundary",
                "d1_test_function_values_boundary",
            ]:
                data[key] = ds.data.to_dataset()

            # Manually copy attributes
            for attr in [
                ("grid_density", lambda x: float(x)),
                ("grid_dimension", lambda x: int(x)),
                ("space_dimensions", lambda x: [str(_) for _ in x]),
                ("test_function_dims", lambda x: [str(_) for _ in x]),
            ]:
                if attr[0] in ds.attrs.keys():

                    data[key].attrs[attr[0]] = attr[1](ds.attrs[attr[0]])

        # Rename data
        data["training_data"] = data["training_data"].rename(
            {"training_data": "boundary_data"}
        )

        # Stack the test function indices into a single multi-index
        for key in [
            "test_function_values",
            "d1_test_function_values",
            "d2_test_function_values",
            "d1_test_function_values_boundary",
            "f_integrated",
        ]:
            data[key] = (
                data[key]
                .stack(tf_idx=data[key].attrs["test_function_dims"])
                .transpose("tf_idx", ...)
            )

        log.info("   All data loaded")

        if not copy_data:
            return data
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
        data["grid_boundary"] = boundary
        training_boundary = (
            boundary
            if boundary_isel is None
            else boundary.isel(base.get_boundary_isel(boundary_isel, grid))
        )
        log.note("   Constructed the grid.")

        # The test functions are only defined on [-1, 1], so a separate grid is used to generate
        # test function values
        tf_space_dict = paramspace.tools.recursive_replace(
            copy.deepcopy(space_dict),
            select_func=lambda d: "extent" in d,
            replace_func=lambda d: d.update(dict(extent=[-1, 1])) or d,
        )

        tf_grid: xr.DataArray = base.construct_grid(tf_space_dict)
        tf_boundary: xr.Dataset = base.get_boundary(tf_grid)

        log.debug("   Evaluating test functions on grid ...")
        test_function_indices = base.construct_grid(
            test_func_dict["num_functions"], lower=1, dtype=int
        )
        test_function_values = base.tf_grid_evaluation(
            tf_grid, test_function_indices, type=test_func_dict["type"], d=0
        )

        log.note("   Evaluated the test functions.")
        data["test_function_values"] = test_function_values.stack(
            tf_idx=test_function_values.attrs["test_function_dims"]
        ).transpose("tf_idx", ...)

        log.debug("   Evaluating test function derivatives on grid ... ")
        d1_test_function_values = base.tf_grid_evaluation(
            tf_grid, test_function_indices, type=test_func_dict["type"], d=1
        )

        data["d1_test_function_values"] = d1_test_function_values.stack(
            tf_idx=test_function_values.attrs["test_function_dims"]
        ).transpose("tf_idx", ...)

        d1_test_function_values_boundary = base.tf_simple_evaluation(
            tf_boundary.sel(variable=tf_grid.attrs["space_dimensions"]),
            test_function_indices,
            type=test_func_dict["type"],
            d=1,
            core_dim="variable",
        )
        data[
            "d1_test_function_values_boundary"
        ] = d1_test_function_values_boundary.stack(
            tf_idx=test_function_values.attrs["test_function_dims"]
        ).transpose(
            "tf_idx", ...
        )

        log.debug("   Evaluating test function second derivatives on grid ... ")
        d2_test_function_values = base.tf_grid_evaluation(
            tf_grid, test_function_indices, type=test_func_dict["type"], d=2
        )
        data["d2_test_function_values"] = d2_test_function_values.stack(
            tf_idx=test_function_values.attrs["test_function_dims"]
        ).transpose("tf_idx", ...)

        log.debug("   Evaluating the external function on the grid ...")
        f_evaluated: xr.DataArray = xr.apply_ufunc(
            forcing, grid, input_core_dims=[["idx"]], vectorize=True
        )
        data["f_evaluated"] = f_evaluated

        log.debug("   Integrating the function over the grid ...")
        f_integrated = base.integrate_xr(f_evaluated, test_function_values)
        data["f_integrated"] = f_integrated.stack(
            tf_idx=f_integrated.attrs["test_function_dims"]
        ).transpose("tf_idx", ...)

        log.debug("   Evaluating the solution on the boundary ...")
        training_data: xr.Dataset = xr.concat(
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
        data["training_data"] = training_data

        log.info("   Generated data.")
    # ------------------------------------------------------------------------------------------------------------------
    # --- Set up chunked dataset to store the state data in ------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    log.info("   Saving data ... ")
    data_group = h5file.create_group("data")

    # ------------------------------------------------------------------------------------------------------------------
    # --- Grid and grid boundary ---------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # Grid
    grid = data["grid"]
    dset_grid = data_group.create_dataset(
        "grid",
        list(grid.sizes.values()),
        maxshape=list(grid.sizes.values()),
        chunks=True,
        compression=3,
    )
    dset_grid.attrs["dim_names"] = [str(_) for _ in list(grid.sizes)]

    # Set attributes
    for idx in list(grid.sizes):
        dset_grid.attrs["coords_mode__" + str(idx)] = "values"
        dset_grid.attrs["coords__" + str(idx)] = grid.coords[idx].data
    dset_grid.attrs.update(grid.attrs)

    # Write data
    dset_grid[
        :,
    ] = grid

    # Grid boundary
    dset_boundary = data_group.create_dataset(
        "grid_boundary",
        list(data["grid_boundary"].sizes.values()),
        maxshape=list(data["grid_boundary"].sizes.values()),
        chunks=True,
        compression=3,
    )
    dset_boundary.attrs["dim_names"] = ["idx", "variable"]

    # Set attributes
    dset_boundary.attrs["coords_mode__idx"] = "trivial"
    dset_boundary.attrs["coords_mode__variable"] = "values"
    dset_boundary.attrs["coords__variable"] = [
        str(_) for _ in data["grid_boundary"].coords["variable"].data
    ]
    dset_boundary.attrs.update(data["grid_boundary"].attrs)

    # Write data
    dset_boundary[
        :,
    ] = data["grid_boundary"].to_array()

    # ------------------------------------------------------------------------------------------------------------------
    # --- Test function values -----------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # Test function values
    test_function_values = data["test_function_values"].unstack()
    dset_test_function_values = data_group.create_dataset(
        "test_function_values",
        test_function_values.shape,
        maxshape=test_function_values.shape,
        chunks=True,
        compression=3,
    )
    dset_test_function_values.attrs["dim_names"] = [
        str(_) for _ in list(test_function_values.sizes)
    ]

    # First derivatives of the test function values
    d1_test_function_values = data["d1_test_function_values"].unstack()
    dset_d1_test_function_values = data_group.create_dataset(
        "d1_test_function_values",
        d1_test_function_values.shape,
        maxshape=d1_test_function_values.shape,
        chunks=True,
        compression=3,
    )
    dset_d1_test_function_values.attrs["dim_names"] = [
        str(_) for _ in list(d1_test_function_values.sizes)
    ]

    # Second derivatives of the test function values
    d2_test_function_values = data["d2_test_function_values"].unstack()
    dset_d2_test_function_values = data_group.create_dataset(
        "d2_test_function_values",
        d2_test_function_values.shape,
        maxshape=d2_test_function_values.shape,
        chunks=True,
        compression=3,
    )
    dset_d2_test_function_values.attrs["dim_names"] = [
        str(_) for _ in list(d2_test_function_values.sizes)
    ]

    # Set attributes
    for idx in list(test_function_values.sizes):
        dset_test_function_values.attrs["coords_mode__" + str(idx)] = "values"
        dset_test_function_values.attrs[
            "coords__" + str(idx)
        ] = test_function_values.coords[idx].data

        dset_d1_test_function_values.attrs["coords_mode__" + str(idx)] = "values"
        dset_d1_test_function_values.attrs[
            "coords__" + str(idx)
        ] = d1_test_function_values.coords[idx].data

        dset_d2_test_function_values.attrs["coords_mode__" + str(idx)] = "values"
        dset_d2_test_function_values.attrs[
            "coords__" + str(idx)
        ] = d2_test_function_values.coords[idx].data
    dset_test_function_values.attrs.update(test_function_values.attrs)
    dset_d1_test_function_values.attrs.update(d1_test_function_values.attrs)
    dset_d2_test_function_values.attrs.update(d2_test_function_values.attrs)

    # Write the data
    dset_test_function_values[
        :,
    ] = test_function_values
    dset_d1_test_function_values[
        :,
    ] = d1_test_function_values
    dset_d2_test_function_values[
        :,
    ] = d2_test_function_values

    # First derivatives of the test function values on the boundary
    d1_test_function_values_boundary = (
        data["d1_test_function_values_boundary"].unstack().to_array()
    )

    dset_d1_test_function_values_boundary = data_group.create_dataset(
        "d1_test_function_values_boundary",
        list(d1_test_function_values_boundary.sizes.values()),
        maxshape=list(d1_test_function_values_boundary.sizes.values()),
        chunks=True,
        compression=3,
    )
    dset_d1_test_function_values_boundary.attrs["dim_names"] = [
        str(_) for _ in list(d1_test_function_values_boundary.sizes)
    ]

    # Set attributes
    for idx in list(d1_test_function_values_boundary.sizes):
        dset_d1_test_function_values_boundary.attrs[
            "coords_mode__" + str(idx)
        ] = "values"
        dset_d1_test_function_values_boundary.attrs[
            "coords__" + str(idx)
        ] = d1_test_function_values_boundary.coords[idx].data

    dset_d1_test_function_values_boundary.attrs.update(
        d1_test_function_values_boundary.attrs
    )

    # Write the data
    dset_d1_test_function_values_boundary[
        :,
    ] = d1_test_function_values_boundary

    # ------------------------------------------------------------------------------------------------------------------
    # --- External forcing ---------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # External function evaluated on the grid
    f_evaluated = data["f_evaluated"]
    dset_f_evaluated = data_group.create_dataset(
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
        dset_f_evaluated.attrs["coords__" + str(idx)] = grid.coords[idx].data
    dset_f_evaluated.attrs.update(f_evaluated.attrs)

    # Write the data
    dset_f_evaluated[
        :,
    ] = f_evaluated

    # Integral of the forcing against the test functions.
    # This dataset is indexed by the test function indices
    f_integrated = data["f_integrated"].unstack()
    dset_f_integrated = data_group.create_dataset(
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

    # ------------------------------------------------------------------------------------------------------------------
    # --- Boundary training data ---------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # Training data: values of the test function on the boundary
    training_data = data["training_data"]
    dset_training_data = data_group.create_dataset(
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

    log.info("   All data saved.")

    return data
