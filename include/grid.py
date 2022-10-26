from typing import Sequence, Union

import numpy as np
import xarray


def construct_grid(
    space_dict: dict, *, lower: int = None, upper: int = None, dtype=None
) -> xarray.DataArray:

    """Constructs a grid from a configuration dictionary, and returns it as a xarray.DataArray.

    :param space_dict: the dictionary of space configurations
    :param lower: (optional) the lower grid value to use if no extent is given
    :param upper: (optional) the upper grid value to use if no extent is given. Defaults to the extent of the grid.
    :param dtype: (optional) the grid value datatype to use
    :return: the grid (xarray.DataArray)

    raises:
        ValueError: for grid dimensions of size greater than 3 (currently not implemented)
    """

    coords, data_vars = {}, {}

    # Create the coordinate dictionary
    for key, entry in space_dict.items():
        l, u = entry.get("extent", (lower, upper))
        u = entry["size"] if u is None else u
        coords[key] = (key, np.linspace(l, u, entry["size"], dtype=dtype))

    # Create the meshgrid. The order of the indices needs to be adapted to the output shape of np.meshgrid.
    indices = list(coords.keys()) + ["idx"]
    if len(indices) == 2:
        data = np.reshape([coords[indices[0]][1]], (-1, 1))

    elif len(indices) == 3:

        x, y = np.meshgrid(coords[indices[0]][1], coords[indices[1]][1])
        indices[0], indices[1] = indices[1], indices[0]
        data = np.stack([x, y], axis=-1)

    elif len(indices) == 4:
        x, y, z = np.meshgrid(
            coords[indices[0]][1], coords[indices[1]][1], coords[indices[2]][1]
        )
        indices[0], indices[1] = indices[1], indices[0]
        data = np.stack([x, y, z], axis=-1)

    else:
        raise ValueError(
            f"Currently not implemented for grid dimensions > 4; got dimension {len(indices)-1}!"
        )

    # Update the DataArray coordinates
    coords.update(dict(idx=("idx", np.arange(1, len(indices), 1))))

    # Add grid dimension as an attribute
    res = xarray.DataArray(coords=coords, data=data, dims=indices)
    res.attrs["grid_dimension"] = len(indices) - 1

    # Calculate grid density for integration
    n_points, volume = np.prod(list(res.sizes.values())) / res.sizes["idx"], 1
    space_dims = []
    for key, range in res.coords.items():
        if key == "idx":
            continue
        volume *= (range.isel({key: -1}) - range.isel({key: 0})).data
        space_dims.append(str(key))

    res.attrs["grid_density"] = volume / n_points
    res.attrs["space_dimensions"] = space_dims

    return res


def get_boundary(grid: xarray.DataArray) -> xarray.Dataset:

    """Extracts the boundary from a grid. Returns the boundary with unit normals associated with each point."""

    # 1D boundary
    if grid.attrs["grid_dimension"] == 1:

        x = grid.attrs["space_dimensions"][0]
        x_0, x_1 = (
            grid.isel({x: 0, "idx": 0}, drop=True).data,
            grid.isel({x: -1, "idx": 0}, drop=True).data,
        )

        return xarray.Dataset(
            coords=dict(idx=("idx", [0, 1]), variable=("variable", [x, "normals_x"])),
            data_vars=dict(boundary_data=(["idx", "variable"], [[x_0, -1], [x_1, +1]])),
            attrs=grid.attrs,
        )

    # 2D boundary
    elif grid.attrs["grid_dimension"] == 2:

        x, y = grid.attrs["space_dimensions"]
        len_x, len_y = len(grid.coords[x].data), len(grid.coords[y].data)

        x_vals = y_vals = n_vals_x = n_vals_y = np.array([])

        # lower boundary
        x_vals = np.append(
            x_vals,
            grid.coords[x].data[slice(None, -1)],
        )
        y_vals = np.append(y_vals, np.repeat(grid.coords[y].data[0], len_x - 1))
        n_vals_x = np.append(n_vals_x, np.repeat(0, len_x - 1))
        n_vals_y = np.append(n_vals_y, np.repeat(-1, len_x - 1))

        # Right boundary
        y_vals = np.append(
            y_vals,
            grid.coords[y].data[slice(None, -1)],
        )
        x_vals = np.append(x_vals, np.repeat(grid.coords[x].data[-1], len_y - 1))
        n_vals_x = np.append(n_vals_x, np.repeat(1, len_y - 1))
        n_vals_y = np.append(n_vals_y, np.repeat(0, len_y - 1))

        # upper boundary (reversed)
        x_vals = np.append(
            x_vals,
            grid.coords[x].data[slice(None, 0, -1)],
        )
        y_vals = np.append(y_vals, np.repeat(grid.coords[y].data[-1], len_x - 1))
        n_vals_x = np.append(n_vals_x, np.repeat(0, len_x - 1))
        n_vals_y = np.append(n_vals_y, np.repeat(1, len_x - 1))

        # Left boundary (reversed)
        y_vals = np.append(
            y_vals,
            grid.coords[y].data[slice(None, 0, -1)],
        )
        x_vals = np.append(x_vals, np.repeat(grid.coords[x].data[0], len_y - 1))
        n_vals_x = np.append(n_vals_x, np.repeat(-1, len_y - 1))
        n_vals_y = np.append(n_vals_y, np.repeat(0, len_y - 1))

        return xarray.Dataset(
            coords=dict(
                idx=("idx", np.arange(0, len(x_vals), 1)),
                variable=("variable", [x, y, "normals_x", "normals_y"]),
            ),
            data_vars=dict(
                boundary_data=(
                    ["idx", "variable"],
                    np.stack([x_vals, y_vals, n_vals_x, n_vals_y], axis=1),
                )
            ),
            attrs=grid.attrs,
        )

    # 3D boundary
    else:
        x, y, z = grid.attrs["space_dimensions"]
        len_x, len_y, len_z = (
            len(grid.coords[x].data),
            len(grid.coords[y].data),
            len(grid.coords[z].data),
        )

        x_vals = y_vals = z_vals = n_vals_x = n_vals_y = n_vals_z = np.array([])

        # Front panel
        for j in range(len_x - 1):
            z_vals = np.append(z_vals, grid.coords[z].data)
            x_vals = np.append(x_vals, np.repeat(grid.coords[x].data[j], len_z))
        y_vals = np.append(
            y_vals, np.repeat(grid.coords[y].data[0], (len_x - 1) * len_z)
        )
        n_vals_x = np.append(n_vals_x, np.repeat(0, (len_x - 1) * len_z))
        n_vals_y = np.append(n_vals_y, np.repeat(-1, (len_x - 1) * len_z))
        n_vals_z = np.append(n_vals_z, np.repeat(0, (len_x - 1) * len_z))

        # Right panel
        for j in range(len_y - 1):
            z_vals = np.append(z_vals, grid.coords[z].data)
            y_vals = np.append(y_vals, np.repeat(grid.coords[y].data[j], len_z))
        x_vals = np.append(
            x_vals, np.repeat(grid.coords[x].data[-1], (len_y - 1) * len_z)
        )
        n_vals_x = np.append(n_vals_x, np.repeat(0, (len_y - 1) * len_z))
        n_vals_y = np.append(n_vals_y, np.repeat(1, (len_y - 1) * len_z))
        n_vals_z = np.append(n_vals_z, np.repeat(0, (len_y - 1) * len_z))

        # Back panel (reversed)
        for j in range(len_x - 1, 0, -1):
            z_vals = np.append(z_vals, grid.coords[z].data)
            x_vals = np.append(x_vals, np.repeat(grid.coords[x].data[j], len_z))
        y_vals = np.append(
            y_vals, np.repeat(grid.coords[y].data[-1], (len_x - 1) * len_z)
        )
        n_vals_x = np.append(n_vals_x, np.repeat(0, (len_x - 1) * len_z))
        n_vals_y = np.append(n_vals_y, np.repeat(+1, (len_x - 1) * len_z))
        n_vals_z = np.append(n_vals_z, np.repeat(0, (len_x - 1) * len_z))

        # Left panel (reversed)
        for j in range(len_y - 1, 0, -1):
            z_vals = np.append(z_vals, grid.coords[z].data)
            y_vals = np.append(y_vals, np.repeat(grid.coords[y].data[j], len_z))
        x_vals = np.append(
            x_vals, np.repeat(grid.coords[x].data[0], (len_y - 1) * len_z)
        )
        n_vals_x = np.append(n_vals_x, np.repeat(0, (len_y - 1) * len_z))
        n_vals_y = np.append(n_vals_y, np.repeat(-1, (len_y - 1) * len_z))
        n_vals_z = np.append(n_vals_z, np.repeat(0, (len_y - 1) * len_z))

        # upper panel
        for j in range(1, len_y - 1):
            x_vals = np.append(x_vals, grid.coords[x].data[1:-1])
            y_vals = np.append(y_vals, np.repeat(grid.coords[y].data[j], len_x - 2))
        z_vals = np.append(
            z_vals, np.repeat(grid.coords[z].data[-1], (len_x - 2) * (len_y - 2))
        )
        n_vals_x = np.append(n_vals_x, np.repeat(0, (len_x - 2) * (len_y - 2)))
        n_vals_y = np.append(n_vals_y, np.repeat(0, (len_x - 2) * (len_y - 2)))
        n_vals_z = np.append(n_vals_z, np.repeat(1, (len_x - 2) * (len_y - 2)))

        # lower panel (reversed)
        for j in range(len_y - 2, 0, -1):
            x_vals = np.append(x_vals, grid.coords[x].data[-2:0:-1])
            y_vals = np.append(y_vals, np.repeat(grid.coords[y].data[j], len_x - 2))
        z_vals = np.append(
            z_vals, np.repeat(grid.coords[z].data[0], (len_x - 2) * (len_y - 2))
        )
        n_vals_x = np.append(n_vals_x, np.repeat(0, (len_x - 2) * (len_y - 2)))
        n_vals_y = np.append(n_vals_y, np.repeat(0, (len_x - 2) * (len_y - 2)))
        n_vals_z = np.append(n_vals_z, np.repeat(-1, (len_x - 2) * (len_y - 2)))

        return xarray.Dataset(
            coords=dict(
                idx=("idx", np.arange(0, len(x_vals), 1)),
                variable=("variable", [x, y, z, "normals_x", "normals_y", "normals_z"]),
            ),
            data_vars=dict(
                boundary_data=(
                    ["idx", "variable"],
                    np.stack(
                        [x_vals, y_vals, z_vals, n_vals_x, n_vals_y, n_vals_z], axis=1
                    ),
                )
            ),
            attrs=grid.attrs,
        )


def get_boundary_isel(
    boundary: xarray.Dataset,
    selection: Union[Sequence[Union[str, slice]], str, slice],
    grid: xarray.DataArray,
) -> xarray.Dataset:
    """Returns a section of the boundary, either indexed by a slice range or by a keyword argument, or by combinations
     thereof. Permissible keyword arguments are:
         - 1-dimensional grid: 'left', 'right'
         - 2-dimensional grid: 'left', 'right', 'lower', 'upper', or combinations
         - 3-dimensional grid: 'left', 'right', 'lower', 'upper', 'front', 'back'
    If a slice of the kind (start, stop, step) is passed, this selection is made on the 'idx' argument of the grid boundary.
    The arguments can also be mixed in a Sequence, e.g.
        - ['left', 'right']
        - [!slice [0, 15], 'upper', 'left'], etc.

    :param boundary: the boundary to slice
    :param selection: (Sequence str or slice) the boundary selection argument
    :param grid: the underlying grid.
    :return: a dictionary containing the selection, passed on to xr.DataArray.isel()
    """
    if isinstance(selection, slice):
        return boundary.isel(dict(idx=selection))

    elif isinstance(selection, str):
        # One-dimensional grid
        if grid.attrs["grid_dimension"] == 1:
            if selection == "left":
                return boundary.isel(dict(idx=0))
            elif selection == "right":
                return boundary.isel(dict(idx=-1))
            else:
                raise ValueError(
                    f"1-dimensional grid has no {selection} boundary! Pass an index range "
                    f"or one of 'left', 'right'."
                )
        elif grid.attrs["grid_dimension"] == 2:
            len_x, len_y = len(grid.coords[grid.attrs["space_dimensions"][0]]), len(
                grid.coords[grid.attrs["space_dimensions"][1]]
            )
            l0, l1 = 0, len_x - 1
            r0, r1 = l1, l1 + len_y - 1
            u0, u1 = r1, r1 + len_x - 1
            le0, le1 = u1, None
            if selection == "lower":
                return boundary.isel(dict(idx=slice(l0, l1)))
            elif selection == "upper":
                return boundary.isel(dict(idx=slice(u0, u1)))
            elif selection == "left":
                return boundary.isel(dict(idx=slice(le0, le1)))
            elif selection == "right":
                return boundary.isel(dict(idx=slice(r0, r1)))
            else:
                raise ValueError(
                    f"2-dimensional grid has no {selection} boundary! Pass an index range "
                    f"or one of 'left', 'right', 'upper', 'lower'."
                )
        elif grid.attrs["grid_dimension"] == 3:
            len_x, len_y, len_z = (
                len(grid.coords[grid.attrs["space_dimensions"][0]]),
                len(grid.coords[grid.attrs["space_dimensions"][1]]),
                len(grid.coords[grid.attrs["space_dimensions"][2]]),
            )
            front_0, front_1 = None, (len_x - 1) * len_z
            right_0, right_1 = front_1, front_1 + (len_y - 1) * len_z
            back_0, back_1 = right_1, right_1 + (len_x - 1) * len_z
            left_0, left_1 = back_1, back_1 + (len_y - 1) * len_z
            upper_0, upper_1 = left_1, left_1 + (len_x - 2) * (len_y - 2)
            lower_0, lower_1 = upper_1, None

            if selection == "front":
                return boundary.isel(dict(idx=slice(front_0, front_1)))
            if selection == "right":
                return boundary.isel(dict(idx=slice(right_0, right_1)))
            if selection == "back":
                return boundary.isel(dict(idx=slice(back_0, back_1)))
            if selection == "left":
                return boundary.isel(dict(idx=slice(left_0, left_1)))
            if selection == "upper":
                return boundary.isel(dict(idx=slice(upper_0, upper_1)))
            if selection == "lower":
                return boundary.isel(dict(idx=slice(lower_0, lower_1)))
            else:
                raise ValueError(
                    f"3-dimensional grid has no {selection} boundary! Pass an index range "
                    f"or one of 'front', 'right', 'back', 'left', 'upper', 'lower'."
                )
    # Combine individual sections of the boundary into one
    elif isinstance(selection, Sequence):
        res = []
        for sel in selection:
            res.append(get_boundary_isel(boundary, sel, grid))
        return xarray.concat(res, dim="idx")
    else:
        raise ValueError(f"Unrecognised boundary selection criterion {selection}!")
