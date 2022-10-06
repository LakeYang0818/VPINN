import h5py as h5
import logging
from os.path import dirname as up
import sys
import xarray as xr
from dantro._import_tools import import_module_from_path

log = logging.getLogger(__name__)

sys.path.append(up(up(__file__)))

base = import_module_from_path(mod_path=up(up(__file__)), mod_str='include')


# ----------------------------------------------------------------------------------------------------------------------
# -- Load or generate grid and training data ---------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def get_data(load_from_file: str,
             space_dict: dict,
             test_func_dict: dict,
             *,
             forcing: callable,
             var_form: int,
             h5file: h5.File) -> dict:

    """Returns the grid and test function data, either by loading it from a file or generating it.
    If generated, data is to written to the output folder

    :param load_from_file: the path to the data file. If none, data is automatically generated
    :param space_dict: the dictionary containing the space configuration
    :param test_func_dict: the dictionary containing the test function configuration
    :param forcing: the external function
    :param var_form: the variational form to use
    :param h5file: the h5file to write data to
    :return: data: a dictionary containing the grid and test function data
    """

    data = {}

    if load_from_file is not None:

        log.info('   Loading data ...')
        data = {}

        with h5.File(load_from_file, "r") as f:
            data['grid'] = f['grid']
            data['test_func_values'] = f['test_function_values']

            if var_form >= 1:
                data['d1test_func_values'] = f['d1_test_function_values']

            if var_form >= 2:
                data['d2test_func_values'] = f['d2_test_function_values']
                data['d1test_func_values_bd'] = f['d1_test_function_values_boundary']

        log.info('   All data loaded')

    else:
        log.info('   Generating data ...')

        log.debug('   Constructing the grid ... ')
        grid: xr.Dataset = base.construct_grid(space_dict)
        data['grid'] = grid

        log.debug('   Evaluating test functions on grid ...')
        test_function_indices = base.construct_grid(
            test_func_dict['num_functions'], lower=1, dtype=int
        )
        test_function_values = base.evaluate_test_functions_on_grid(grid, test_function_indices,
                                                                    type=test_func_dict['type'], d=0)
        data['test_func_values'] = test_function_values

        log.debug('   Evaluating test function derivatives on grid ... ')
        data['d1test_func_values'] = base.evaluate_test_functions_on_grid(grid, test_function_indices,
                                                                          type=test_func_dict['type'], d=1)

        log.debug('   Evaluating test function second derivatives on grid ... ')
        data['d2test_func_values'] = base.evaluate_test_functions_on_grid(grid, test_function_indices,
                                                                          type=test_func_dict['type'], d=2)

        log.debug('   Evaluating the external function on the grid ...')
        f_evaluated: xr.DataArray = xr.apply_ufunc(forcing, grid, input_core_dims=[['idx']], vectorize=True)

        data['f_evaluated'] = f_evaluated

        # --- Set up chunked dataset to store the state data in --------------------------------------------------------
        data_group = h5file.create_group('data')

        # Store the grid
        dset_grid = data_group.create_dataset(
            "grid",
            list(grid.sizes.values()),
            maxshape=list(grid.sizes.values()),
            chunks=True,
            compression=3,
        )
        dset_grid.attrs['dim_names'] = list(grid.sizes)

        # Set attributes
        for idx in list(grid.sizes):
            dset_grid.attrs['coords_mode__' + str(idx)] = 'values'
            dset_grid.attrs['coords__' + str(idx)] = grid.coords[idx].data

        dset_grid[:, ] = grid

        # Store the test function values
        dset_test_func_vals = data_group.create_dataset(
            "test_function_values",
            test_function_values.shape,
            maxshape=test_function_values.shape,
            chunks=True,
            compression=3,
        )
        dset_test_func_vals.attrs['dim_names'] = list(test_function_values.sizes)

        # Store the first derivatives of the test function values
        dset_d1_test_func_vals = data_group.create_dataset(
            "d1_test_function_values",
            test_function_values.shape,
            maxshape=test_function_values.shape,
            chunks=True,
            compression=3,
        )
        dset_d1_test_func_vals.attrs['dim_names'] = list(test_function_values.sizes)

        # Store the second derivatives of the test function values
        dset_d2_test_func_vals = data_group.create_dataset(
            "d2_test_function_values",
            test_function_values.shape,
            maxshape=test_function_values.shape,
            chunks=True,
            compression=3,
        )
        dset_d2_test_func_vals.attrs['dim_names'] = list(test_function_values.sizes)

        for idx in list(test_function_values.sizes):
            dset_test_func_vals.attrs['coords_mode__' + str(idx)] = 'values'
            dset_test_func_vals.attrs['coords__' + str(idx)] = test_function_values.coords[idx].data

            dset_d1_test_func_vals.attrs['coords_mode__' + str(idx)] = 'values'
            dset_d1_test_func_vals.attrs['coords__' + str(idx)] = test_function_values.coords[idx].data

            dset_d2_test_func_vals.attrs['coords_mode__' + str(idx)] = 'values'
            dset_d2_test_func_vals.attrs['coords__' + str(idx)] = test_function_values.coords[idx].data

        dset_test_func_vals[:, ] = test_function_values
        dset_d1_test_func_vals[:, ] = data['d1test_func_values']
        dset_d2_test_func_vals[:, ] = data['d2test_func_values']

        # Store the function evaluated on the grid
        dset_f_evaluated = data_group.create_dataset(
            "f_evaluated",
            list(f_evaluated.sizes.values()),
            maxshape=list(f_evaluated.sizes.values()),
            chunks=True,
            compression=3,
        )
        dset_f_evaluated.attrs['dim_names'] = list(f_evaluated.sizes)

        for idx in list(f_evaluated.sizes):
            dset_f_evaluated.attrs['coords_mode__' + str(idx)] = 'values'
            dset_f_evaluated.attrs['coords__' + str(idx)] = grid.coords[idx].data

        dset_f_evaluated[:, ] = f_evaluated

        # Integrate the external function integrated over the grid
        log.debug('   Integrating the function over the grid ...')
        f_integrated = base.integrate_xr(f_evaluated, test_function_values)

        dset_f_integrated = data_group.create_dataset(
            "f_integrated",
            list(f_integrated.sizes.values()),
            maxshape=list(f_integrated.sizes.values()),
            chunks=True,
            compression=3,
        )
        dset_f_integrated.attrs['dim_names'] = list(f_integrated.sizes)

        for idx in list(f_integrated.sizes):
            dset_f_integrated.attrs['coords_mode__' + str(idx)] = 'values'
            dset_f_integrated.attrs['coords__' + str(idx)] = f_integrated.coords[idx].data
        dset_f_integrated[:, ] = f_integrated

        data['f_integrated'] = f_integrated

        log.info('   All data generated and saved.')

    return data
