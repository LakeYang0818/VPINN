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
            coords=dict(idx=("idx", [0, 1]), variable=("variable", [x, "n"])),
            data_vars=dict(data=(["idx", "variable"], [[x_0, -1], [x_1, +1]])),
            attrs=grid.attrs,
        )

    # 2D boundary
    elif grid.attrs["grid_dimension"] == 2:

        x, y = grid.attrs["space_dimensions"]
        len_x, len_y = len(grid.coords[x].data), len(grid.coords[y].data)

        x_vals = y_vals = n_vals = np.array([])

        # Bottom and top boundaries
        for i in [0, -1]:
            x_vals = np.append(
                x_vals,
                grid.coords[x].data[slice(1, None) if i == 0 else slice(None, -1)],
            )
            y_vals = np.append(y_vals, np.repeat(grid.coords[y].data[i], len_x - 1))
            n_vals = np.append(n_vals, np.repeat(1 if i == 0 else i, len_x - 1))

        # Left and right boundaries
        for i in [0, -1]:
            y_vals = np.append(
                y_vals,
                grid.coords[y].data[slice(1, -1) if i == 0 else slice(None, None)],
            )
            x_vals = np.append(x_vals, np.repeat(grid.coords[x].data[i], len_y - 1))
            n_vals = np.append(n_vals, np.repeat(1 if i == -1 else -1, len_y - 1))

        return xarray.Dataset(
            coords=dict(
                idx=("idx", np.arange(0, len(x_vals), 1)),
                variable=("variable", [x, y, "n"]),
            ),
            data_vars=dict(
                data=(["idx", "variable"], np.stack([x_vals, y_vals, n_vals], axis=1))
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

        x_vals = y_vals = z_vals = n_vals = np.array([])

        # Bottom and top boundaries
        for i in [0, -1]:
            for j in range(len_y):
                x_vals = np.append(x_vals, grid.coords[x].data)
                y_vals = np.append(y_vals, np.repeat(grid.coords[y].data[j], len_x))
            z_vals = np.append(
                z_vals, np.repeat(grid.coords[z].data[i], (len_x) * (len_y))
            )
            n_vals = np.append(n_vals, np.repeat(1 if i == 0 else i, (len_x) * (len_y)))

        # Left and right boundaries
        for i in [0, -1]:
            for j in range(1, len_z - 1):
                y_vals = np.append(y_vals, grid.coords[y].data)
                z_vals = np.append(z_vals, np.repeat(grid.coords[z].data[j], len_y))
            x_vals = np.append(
                x_vals, np.repeat(grid.coords[x].data[i], (len_y) * (len_z - 2))
            )
            n_vals = np.append(
                n_vals, np.repeat(1 if i == -1 else -1, (len_y) * (len_z - 2))
            )

        # Front and back boundaries
        for i in [0, -1]:
            for j in range(1, len_x - 1):
                z_vals = np.append(z_vals, grid.coords[z].data[1:-1])
                x_vals = np.append(x_vals, np.repeat(grid.coords[x].data[j], len_z - 2))
            y_vals = np.append(
                y_vals, np.repeat(grid.coords[y].data[i], (len_x - 2) * (len_z - 2))
            )
            n_vals = np.append(
                n_vals, np.repeat(1 if i == -1 else -1, (len_x - 2) * (len_z - 2))
            )

        return xarray.Dataset(
            coords=dict(
                idx=("idx", np.arange(0, len(x_vals), 1)),
                variable=("variable", [x, y, z, "n"]),
            ),
            data_vars=dict(
                data=(
                    ["idx", "variable"],
                    np.stack([x_vals, y_vals, z_vals, n_vals], axis=1),
                )
            ),
            attrs=grid.attrs,
        )


# TODO
def get_interior():
    pass


# def rescale_grid(grid: Grid, *, new_domain) -> Grid:
#     """ Rescales a grid to a new domain.
#
#     :param grid: the grid to rescale
#     :param new_domain: the domain to which to scale the grid
#     :return: the rescaled grid
#
#     :raises ValueError: if a grid is being rescaled to a new domain whose dimension does not match the
#         grid dimension.
#     """
#
#
#     if grid.dim == 1:
#         if np.shape(new_domain) != (2,):
#             raise ValueError(f"Cannot rescale 1d grid to {len(np.shape(new_domain))}-dimensional domain!")
#
#         return Grid(x=(new_domain[1] - new_domain[0]) * (grid.x - grid.x[0]) / (grid.x[-1] - grid.x[0]) + new_domain[0],
#                     as_tensor=grid.is_tensor, dtype=grid.dtype, requires_grad=grid.requires_grad, requires_normals=(grid.normals is not None))
#
#     elif grid.dim == 2:
#         if np.shape(new_domain) != (2, 2):
#             raise ValueError(f"Cannot rescale 2d grid to {len(np.shape(new_domain))}-dimensional domain!")
#
#         return Grid(
#             x=(new_domain[0][1] - new_domain[0][0]) * (grid.x - grid.x[0]) / (grid.x[-1] - grid.x[0]) + new_domain[0][
#                 0],
#             y=(new_domain[1][1] - new_domain[1][0]) * (grid.y - grid.y[0]) / (grid.y[-1] - grid.y[0]) + new_domain[1][
#                 0],
#             as_tensor=grid.is_tensor, dtype=grid.dtype, requires_grad=grid.requires_grad, requires_normals=(grid.normals is not None))
