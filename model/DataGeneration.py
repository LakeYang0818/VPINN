import copy
import logging
import sys
from os.path import dirname as up
from typing import Sequence, Union

import h5py as h5
import paramspace
import xarray as xr
from dantro._import_tools import import_module_from_path

import utopya.eval.datamanager

log = logging.getLogger(__name__)

sys.path.append(up(up(__file__)))

base = import_module_from_path(mod_path=up(up(__file__)), mod_str="include")


def load_grid_tf_data(
    data_dir: str,
    load_cfg: dict,
) -> dict:

    """Loads grid and test function data from an h5 file. If a test function subselection is specified,
    selects a subset of test functions.

    :param data_dir: (str) the directory containing the h5 file
    :param load_cfg: (dict) load settings, passed to :pyfunc:utopya.eval.datamanager.DataManager.load
    :return: a dictionary containing the grid and test function data
    """

    log.info("   Loading data ...")

    data = {}

    # A subset of test functions can be selected to reduce compute time
    tf_sel_dict = {
        key: val for key, val in load_cfg.pop("test_function_subset", {}).items()
    }

    dm = utopya.eval.datamanager.DataManager(data_dir=data_dir, out_dir=False)

    dm.load(
        "VPINN_data",
        loader=load_cfg.pop("loader", "hdf5"),
        glob_str=load_cfg.pop("glob_str", "*.h5"),
        print_tree=load_cfg.pop("print_tree", False),
        **load_cfg,
    )

    for key in list(dm["VPINN_data"]["data"]["grid_test_function_data"].keys()):

        # Get the dataset and convert to xr.DataArray
        ds = dm["VPINN_data"]["data"]["grid_test_function_data"][key]
        data[key] = ds.data

        # These entries are xr.Datasets, rather than xr.DataArrays
        if key in [
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

    # Stack the test function indices into a single multi-index
    for key in [
        "test_function_values",
        "d1_test_function_values",
        "d2_test_function_values",
        "d1_test_function_values_boundary",
    ]:
        data[key] = (
            data[key]
            .sel(tf_sel_dict)
            .stack(tf_idx=data[key].attrs["test_function_dims"])
            .transpose("tf_idx", ...)
        )

    data["grid_boundary"] = data["grid_boundary"].rename(
        {"grid_boundary": "boundary_data"}
    )

    log.info("   Data loaded.")

    return data


def generate_grid_tf_data(
    space_dict: dict,
    test_func_dict: dict,
) -> dict:

    """Generates grid and test function data.

    :param space_dict: the configuration for the space grid
    :param test_func_dict: the configuration for the test function grid
    :return: a dictionary containing the grid and test function data
    """
    data = {}

    if len(space_dict) != len(test_func_dict["num_functions"]):
        raise ValueError(
            f"Space and test function dimensions do not match! "
            f"Got {len(space_dict)} and {len(test_func_dict['num_functions'])}."
        )

    log.info("   Generating grid and test function data ...")

    log.debug("   Constructing the grid ... ")
    grid: xr.DataArray = base.construct_grid(space_dict)
    boundary: xr.Dataset = base.get_boundary(grid)
    data["grid"] = grid
    data["grid_boundary"] = boundary

    # The test functions are only defined on [-1, 1], so a separate grid is used to generate
    # test function values
    log.debug("   Constructing the test function grid ...")
    tf_space_dict = paramspace.tools.recursive_replace(
        copy.deepcopy(space_dict),
        select_func=lambda d: "extent" in d,
        replace_func=lambda d: d.update(dict(extent=[-1, 1])) or d,
    )

    tf_grid: xr.DataArray = base.construct_grid(tf_space_dict)
    tf_boundary: xr.Dataset = base.get_boundary(tf_grid)

    # Evaluate the test functions on the grid, using fast grid evaluation
    log.debug("   Evaluating test functions on grid ...")
    test_function_indices = base.construct_grid(
        test_func_dict["num_functions"], lower=1, dtype=int
    )
    test_function_values = base.tf_grid_evaluation(
        tf_grid, test_function_indices, type=test_func_dict["type"], d=0
    )
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

    log.debug("   Evaluating test function second derivatives on grid ... ")
    d2_test_function_values = base.tf_grid_evaluation(
        tf_grid, test_function_indices, type=test_func_dict["type"], d=2
    )
    data["d2_test_function_values"] = d2_test_function_values.stack(
        tf_idx=test_function_values.attrs["test_function_dims"]
    ).transpose("tf_idx", ...)

    # Evaluate the test function derivatives on the grid boundary
    d1_test_function_values_boundary = base.tf_simple_evaluation(
        tf_boundary.sel(variable=tf_grid.attrs["space_dimensions"]),
        test_function_indices,
        type=test_func_dict["type"],
        d=1,
        core_dim="variable",
    )
    data["d1_test_function_values_boundary"] = d1_test_function_values_boundary.stack(
        tf_idx=test_function_values.attrs["test_function_dims"]
    ).transpose("tf_idx", ...)

    log.info("   Generated grid and test function data.")

    return data


def save_grid_tf_data(data: dict, h5file: h5.File):

    """Saves the grid and test function data to a h5 file

    :param data: the dictionary containing the data
    :param h5file: the h5.File to save the data to
    """

    log.info("   Saving data ... ")
    data_group = h5file.create_group("grid_test_function_data")

    # Initialise grid dataset
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

    # Initialise test function values datasets
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

    log.info("   All data saved.")


def get_grid_tf_data(
    load_cfg: dict,
    space_dict: dict,
    test_func_dict: dict,
    h5file: h5.File,
) -> dict:

    """Returns the grid and test function data, either by loading it from a file or generating it.
    If generated, data is written to the output folder

    :param load_cfg: load configuration, containing the path to the data file. If the path is None, data is
         automatically generated. If the ``copy_data`` entry is true, data will be copied and written to the
         output directory; however, this is false by default to save disk space.
         The configuration also contains further kwargs, passed to the loader.
    :param space_dict: the dictionary containing the space configuration
    :param test_func_dict: the dictionary containing the test function configuration
    :param h5file: the h5file to write data to. A new group is added to this file.

    :return: data: a dictionary containing the grid and test function data
    """

    # The directory from which to load data
    data_dir = load_cfg.pop("data_dir", None)

    # Load data
    if data_dir is not None:

        # If data is loaded, determine whether to copy that data to the new output directory.
        # This is false by default to save disk storage
        copy_data = load_cfg.pop("copy_data", False)

        data = load_grid_tf_data(data_dir, load_cfg)

        # If data is not copied to the new output folder, return data
        if not copy_data:
            return data

    # Generate data
    else:

        data = generate_grid_tf_data(space_dict, test_func_dict)

    # Save the data and return
    save_grid_tf_data(data, h5file)

    return data


def get_training_data(
    *,
    func: callable,
    grid: xr.DataArray,
    boundary: xr.Dataset,
    boundary_isel: Union[Sequence[Union[str, slice]], str, slice, None],
) -> dict:

    """Obtains the training data, given by the boundary conditions on a specified grid boundary.

    :param func: the function to evaluate on the boundary
    :param grid: the grid data
    :param boundary: the grid boundary
    :param boundary_isel: the boundary selection, i.e. which part of the boundary to use as training data
    :return: a dictionary containing the training data
    """

    # Get the training boundary
    log.debug("   Obtaining the training boundary ...")
    training_boundary = (
        boundary
        if boundary_isel is None
        else base.get_boundary_isel(boundary, boundary_isel, grid)
    )

    # Evaluate the function on the training boundary
    log.debug("   Evaluating the solution on the boundary ...")
    training_data: xr.Dataset = xr.concat(
        [
            training_boundary,
            xr.apply_ufunc(
                func,
                training_boundary.sel(variable=grid.attrs["space_dimensions"]),
                input_core_dims=[["variable"]],
                vectorize=True,
            ).assign_coords(variable=("variable", ["u"])),
        ],
        dim="variable",
    )

    return dict(training_data=training_data)


def get_forcing_data(
    *,
    func: callable,
    grid: xr.DataArray,
    test_function_values: xr.DataArray,
) -> dict:

    """Integrates a function against an xr.DataArray of test functions.

    :param func: the function to integrate
    :param grid: the grid
    :param test_function_values: the test function values
    :return: a dictionary containing the function integrated
    """

    log.debug("   Evaluating the external function on the grid ...")
    f_evaluated: xr.DataArray = xr.apply_ufunc(
        func, grid, input_core_dims=[["idx"]], vectorize=True
    )

    log.debug("   Integrating the function over the grid ...")
    f_integrated: xr.DataArray = base.integrate_xr(
        f_evaluated, test_function_values
    ).transpose("tf_idx", ...)

    return dict(f_evaluated=f_evaluated, f_integrated=f_integrated)
