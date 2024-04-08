import numpy as np

def trilinear_interpolation(grid_x, grid_y, grid_z, values, x, y, z):
    xl = grid_x[0]
    xu = grid_x[-1]
    yl = grid_y[0]
    yu = grid_y[-1]
    zl = grid_z[0]
    zu = grid_z[-1]
    
    # Find indices of the bounding grid points
    x_idx = np.searchsorted(grid_x, x) - 1
    y_idx = np.searchsorted(grid_y, y) - 1
    z_idx = np.searchsorted(grid_z, z) - 1

    # Check if the point is within the grid
    if x_idx < 0 or x_idx >= len(grid_x) - 1 or y_idx < 0 or y_idx >= len(grid_y) - 1 or z_idx < 0 or z_idx >= len(grid_z) - 1:
        dx, dy, dz = distance_to_bounding_box(x, y, z, xl, xu, yl, yu, zl, zu)
        return (dx**2 + dy**2 + dz**2)**0.5

    # Coordinates of the eight nearest grid points
    x0, x1 = grid_x[x_idx], grid_x[x_idx + 1]
    y0, y1 = grid_y[y_idx], grid_y[y_idx + 1]
    z0, z1 = grid_z[z_idx], grid_z[z_idx + 1]

    # Perform the interpolations
    c00 = values[x_idx, y_idx, z_idx] * (1 - (x - x0) / (x1 - x0)) + values[x_idx + 1, y_idx, z_idx] * (x - x0) / (x1 - x0)
    c01 = values[x_idx, y_idx, z_idx + 1] * (1 - (x - x0) / (x1 - x0)) + values[x_idx + 1, y_idx, z_idx + 1] * (x - x0) / (x1 - x0)
    c10 = values[x_idx, y_idx + 1, z_idx] * (1 - (x - x0) / (x1 - x0)) + values[x_idx + 1, y_idx + 1, z_idx] * (x - x0) / (x1 - x0)
    c11 = values[x_idx, y_idx + 1, z_idx + 1] * (1 - (x - x0) / (x1 - x0)) + values[x_idx + 1, y_idx + 1, z_idx + 1] * (x - x0) / (x1 - x0)
    
    c0 = c00 * (1 - (y - y0) / (y1 - y0)) + c10 * (y - y0) / (y1 - y0)
    c1 = c01 * (1 - (y - y0) / (y1 - y0)) + c11 * (y - y0) / (y1 - y0)
    
    c = c0 * (1 - (z - z0) / (z1 - z0)) + c1 * (z - z0) / (z1 - z0)

    return c

def distance_to_bounding_box(x, y, z, xl, xu, yl, yu, zl, zu):
    """Compute the shortest distance of a point to the bounding box."""
    dx = max(xl - x, 0, x - xu)
    dy = max(yl - y, 0, y - yu)
    dz = max(zl - z, 0, z - zu)
    return dx, dy, dz