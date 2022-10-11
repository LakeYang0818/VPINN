from typing import Union

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
            data_vars=dict(data=(["idx", "variable"], [[x_0, -1], [x_1, +1]])),
            attrs=grid.attrs,
        )

    # 2D boundary
    elif grid.attrs["grid_dimension"] == 2:

        x, y = grid.attrs["space_dimensions"]
        len_x, len_y = len(grid.coords[x].data), len(grid.coords[y].data)

        x_vals = y_vals = n_vals_x = n_vals_y = np.array([])

        # Bottom and top boundaries
        for i in [0, -1]:
            x_vals = np.append(
                x_vals,
                grid.coords[x].data[slice(1, None) if i == 0 else slice(None, -1)],
            )
            y_vals = np.append(y_vals, np.repeat(grid.coords[y].data[i], len_x - 1))
            n_vals_x = np.append(n_vals_x, np.repeat(0, len_x - 1))
            n_vals_y = np.append(n_vals_y, np.repeat(-1 if i == 0 else 1, len_x - 1))

        # Left and right boundaries
        for i in [0, -1]:
            y_vals = np.append(
                y_vals,
                grid.coords[y].data[slice(1, -1) if i == 0 else slice(None, None)],
            )
            x_vals = np.append(x_vals, np.repeat(grid.coords[x].data[i], len_y - 1))
            n_vals_x = np.append(n_vals_x, np.repeat(-1 if i == 0 else 1, len_y - 1))
            n_vals_y = np.append(n_vals_y, np.repeat(0, len_y - 1))

        return xarray.Dataset(
            coords=dict(
                idx=("idx", np.arange(0, len(x_vals), 1)),
                variable=("variable", [x, y, "normals_x", "normals_y"]),
            ),
            data_vars=dict(
                data=(
                    ["idx", "variable"],
                    np.stack([x_vals, y_vals, n_vals_x, n_vals_y], axis=1),
                )
            ),
            attrs=grid.attrs,
        )

    else:
        x, y, z = grid.attrs["space_dimensions"]
        len_x, len_y, len_z = (
            len(grid.coords[x].data),
            len(grid.coords[y].data),
            len(grid.coords[z].data),
        )

        x_vals = y_vals = z_vals = n_vals_x = n_vals_y = n_vals_z = np.array([])

        # Bottom and top boundaries
        for i in [0, -1]:
            for j in range(len_y):
                x_vals = np.append(x_vals, grid.coords[x].data)
                y_vals = np.append(y_vals, np.repeat(grid.coords[y].data[j], len_x))
            z_vals = np.append(
                z_vals, np.repeat(grid.coords[z].data[i], (len_x) * (len_y))
            )
            n_vals_x = np.append(n_vals_x, np.repeat(0, (len_x) * (len_y)))
            n_vals_y = np.append(n_vals_y, np.repeat(0, (len_x) * (len_y)))
            n_vals_z = np.append(
                n_vals_z, np.repeat(-1 if i == 0 else 1, (len_x) * (len_y))
            )

        # Left and right boundaries
        for i in [0, -1]:
            for j in range(1, len_z - 1):
                y_vals = np.append(y_vals, grid.coords[y].data)
                z_vals = np.append(z_vals, np.repeat(grid.coords[z].data[j], len_y))
            x_vals = np.append(
                x_vals, np.repeat(grid.coords[x].data[i], (len_y) * (len_z - 2))
            )
            n_vals_x = np.append(
                n_vals_x, np.repeat(-1 if i == 0 else 1, (len_x) * (len_y))
            )
            n_vals_y = np.append(n_vals_y, np.repeat(0, (len_x) * (len_y)))
            n_vals_z = np.append(n_vals_z, np.repeat(0, (len_x) * (len_y)))

        # Front and back boundaries
        for i in [0, -1]:
            for j in range(1, len_x - 1):
                z_vals = np.append(z_vals, grid.coords[z].data[1:-1])
                x_vals = np.append(x_vals, np.repeat(grid.coords[x].data[j], len_z - 2))
            y_vals = np.append(
                y_vals, np.repeat(grid.coords[y].data[i], (len_x - 2) * (len_z - 2))
            )
            n_vals_x = np.append(n_vals_x, np.repeat(0, (len_x) * (len_y)))
            n_vals_y = np.append(
                n_vals_y, np.repeat(-1 if i == 0 else 1, (len_x) * (len_y))
            )
            n_vals_z = np.append(n_vals_z, np.repeat(0, (len_x) * (len_y)))

        return xarray.Dataset(
            coords=dict(
                idx=("idx", np.arange(0, len(x_vals), 1)),
                variable=("variable", [x, y, z, "normals_x", "normals_y", "normals_z"]),
            ),
            data_vars=dict(
                data=(
                    ["idx", "variable"],
                    np.stack(
                        [x_vals, y_vals, z_vals, n_vals_x, n_vals_y, n_vals_z], axis=1
                    ),
                )
            ),
            attrs=grid.attrs,
        )


def get_boundary_isel(selection: Union[str, tuple], grid: xarray.DataArray) -> dict:
    """Returns a section of the boundary, either indexed by a range or by a keyword argument. Permissible arguments
    are:
         - 1-dimensional grid: 'left', 'right'
         - 2-dimensional grid: 'left', 'right', 'lower', 'upper'
         - 3-dimensional grid: 'left', 'right', 'lower', 'upper', 'front', 'back'
    If a tuple of the kind (start, stop) is passed, this selection is made on the 'idx' argument of the grid boundary.

    :param selection: (str or tuple) the boundary selection argument
    :param grid: the underlying grid.
    :return: a dictionary containing the selection, passed on to xr.DataArray.isel()
    """
    if isinstance(selection, tuple):
        return dict(idx=slice(selection[0], selection[1]))

    elif isinstance(selection, str):
        # One-dimensional grid
        if grid.attrs["grid_dimension"] == 1:
            if selection == "left":
                return dict(idx=0)
            elif selection == "right":
                return dict(idx=-1)
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
            u0, u1 = l1, l1 + len_x - 1
            le0, le1 = u1, u1 + len_y - 1
            r0, r1 = le1, None
            if selection == "lower":
                return dict(idx=slice(l0, l1))
            elif selection == "upper":
                return dict(idx=slice(u0, u1))
            elif selection == "left":
                return dict(idx=slice(le0, le1))
            elif selection == "right":
                return dict(idx=slice(r0, r1))
            else:
                raise ValueError(
                    f"2-dimensional grid has no {selection} boundary! Pass an index range "
                    f"or one of 'left', 'right', 'top', 'bottom'."
                )
        elif grid.attrs["grid_dimension"] == 3:
            len_x, len_y, len_z = (
                len(grid.coords[grid.attrs["space_dimensions"][0]]),
                len(grid.coords[grid.attrs["space_dimensions"][1]]),
                len(grid.coords[grid.attrs["space_dimensions"][2]]),
            )
            l0, l1 = 0, len_x * len_y
            u0, u1 = l1, 2 * l1
            le0, le1 = u1, u1 + len_y * (len_z - 2)
            r0, r1 = le1, le1 + len_y * (len_z - 2)
            f0, f1 = r1, r1 + (len_x - 2) * (len_z - 2)
            b0, b1 = f1, None
            if selection == "lower":
                return dict(idx=slice(l0, l1))
            elif selection == "upper":
                return dict(idx=slice(u0, u1))
            elif selection == "left":
                return dict(idx=slice(le0, le1))
            elif selection == "right":
                return dict(idx=slice(r0, r1))
            elif selection == "front":
                return dict(idx=slice(f0, f1))
            elif selection == "back":
                return dict(idx=slice(b0, b1))
            else:
                raise ValueError(
                    f"2-dimensional grid has no {selection} boundary! Pass an index range "
                    f"or one of 'left', 'right', 'top', 'bottom'."
                )
    else:
        raise ValueError(f"Unrecognised boundary selection criterion {selection}!")
