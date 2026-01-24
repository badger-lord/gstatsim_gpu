# neighbors_gpu.py

import cupy as cp
import math

def _cond_coords_from_mask(xx, yy, grid, cond_msk=None):
    if cond_msk is None:
        cond_msk = ~cp.isnan(grid)
    cond_ii, cond_jj = cp.where(cond_msk)
    if cond_ii.size == 0:
        return cp.empty((0, 2)), cp.empty((0,)), cond_ii, cond_jj
    cond_x = xx[cond_ii, cond_jj]
    cond_y = yy[cond_ii, cond_jj]
    vals = grid[cond_ii, cond_jj]
    coords = cp.stack([cond_x, cond_y], axis=1)
    return coords, vals, cond_ii, cond_jj

def _sector_indices(dx, dy, n_sectors=8):
    """
    Return sector indices [0..n_sectors-1] for vectors (dx,dy).
    Sector 0 centered at angle -pi to -pi+2pi/n.
    """
    angles = cp.arctan2(dy, dx)  # [-pi, pi]
    # Map to [0, 2pi)
    angles = cp.where(angles < 0, angles + 2 * cp.pi, angles)
    sector = cp.floor(angles / (2 * cp.pi / n_sectors)).astype(cp.int32)
    sector = cp.clip(sector, 0, n_sectors - 1)
    return sector

def batch_neighbors_distance_based(batch_inds, ii, jj, xx, yy, grid, cond_msk,
                                 radius, num_points, max_memory_gb=20.0,
                                 use_sector_balance=False, n_sectors=8):
    """
    OPTIMIZED neighbor search using a localized, vectorized approach.
    Avoids computing a full distance matrix by searching only within a
    fixed-size window around each target point.
    """
    B = batch_inds.shape[0]
    K = int(num_points)
    grid_rows, grid_cols = grid.shape

    if B == 0:
        return cp.empty((0, K, 5), dtype=cp.float64), cp.zeros(0, dtype=cp.int32)

    # --- 1. Define Search Window ---
    # Calculate grid spacing (assuming uniform grid)
    dx = cp.abs(xx[0, 1] - xx[0, 0])
    # Window half-width in grid cells
    W = int(cp.ceil(radius / dx).get())

    # Create stencil of relative indices for the search window
    di, dj = cp.meshgrid(cp.arange(-W, W + 1, dtype=cp.int32),
                         cp.arange(-W, W + 1, dtype=cp.int32),
                         indexing='ij')
    
    # Filter stencil to a circle to reduce candidate points by ~21%
    stencil_dist_sq = di.ravel()**2 + dj.ravel()**2
    keep_mask = stencil_dist_sq <= W**2
    di_flat = di.ravel()[keep_mask]
    dj_flat = dj.ravel()[keep_mask]
    window_size = di_flat.size

    # --- 2. Gather Local Data for the Batch ---
    # Get i, j coordinates for the batch points
    batch_i = batch_inds[:, 0]
    batch_j = batch_inds[:, 1]
    
    # Generate absolute indices for the search window of each batch point
    # Shapes: (B, 1) + (1, window_size) -> (B, window_size)
    window_i = batch_i[:, None] + di_flat[None, :]
    window_j = batch_j[:, None] + dj_flat[None, :]

    # Clip indices to be within grid bounds
    cp.clip(window_i, 0, grid_rows - 1, out=window_i)
    cp.clip(window_j, 0, grid_cols - 1, out=window_j)

    # --- 3. Find Neighbors in Local Windows ---
    local_vals = grid[window_i, window_j]
    local_cond = cond_msk[window_i, window_j]
    is_valid_neighbor = local_cond & ~cp.isnan(local_vals)
    
    # Get coordinates for target points
    target_x = xx[batch_i, batch_j]
    target_y = yy[batch_i, batch_j]

    # Calculate distances only to local points
    local_x = xx[window_i, window_j]
    local_y = yy[window_i, window_j]
    dist_sq = (target_x[:, None] - local_x)**2 + (target_y[:, None] - local_y)**2
    
    # Use a large value (infinity) for invalid points for argpartition
    dists = cp.sqrt(dist_sq)
    masked_dists = cp.where(is_valid_neighbor & (dists <= radius), dists, cp.inf)

    # --- 4. Partition and Sort to Get Top K ---
    K_eff = min(K, window_size)
    partition_idx = cp.argpartition(masked_dists, K_eff - 1, axis=1)[:, :K_eff]

    row_selector = cp.arange(B)[:, None]
    top_k_dists = cp.take_along_axis(masked_dists, partition_idx, axis=1)
    
    sort_order = cp.argsort(top_k_dists, axis=1)
    
    final_window_indices = cp.take_along_axis(partition_idx, sort_order, axis=1)

    # --- 5. Assemble Output Array ---
    neigh = cp.full((B, K, 5), cp.nan, dtype=cp.float64)

    final_x = cp.take_along_axis(local_x, final_window_indices, axis=1)
    final_y = cp.take_along_axis(local_y, final_window_indices, axis=1)
    final_vals = cp.take_along_axis(local_vals, final_window_indices, axis=1)
    final_ii = cp.take_along_axis(window_i, final_window_indices, axis=1)
    final_jj = cp.take_along_axis(window_j, final_window_indices, axis=1)
    final_sorted_dists = cp.take_along_axis(top_k_dists, sort_order, axis=1)

    # Mask out invalid neighbors (where distance was inf)
    valid_mask = cp.isfinite(final_sorted_dists)

    # Assign data to the output array up to K_eff
    neigh[:, :K_eff, 0] = cp.where(valid_mask, final_x, cp.nan)
    neigh[:, :K_eff, 1] = cp.where(valid_mask, final_y, cp.nan)
    neigh[:, :K_eff, 2] = cp.where(valid_mask, final_vals, cp.nan)
    neigh[:, :K_eff, 3] = cp.where(valid_mask, final_ii.astype(cp.float64), cp.nan)
    neigh[:, :K_eff, 4] = cp.where(valid_mask, final_jj.astype(cp.float64), cp.nan)

    nb_counts = cp.sum(valid_mask, axis=1).astype(cp.int32)
    
    return neigh, nb_counts

def make_circle_stencil_gpu_safe(x, rad, max_memory_gb=1.0):
    dx = cp.abs(x[1] - x[0])
    if dx == 0:
        return None, None, None
    ncells = int(cp.ceil(rad / dx).get())
    x_stencil = cp.linspace(-rad, rad, 2 * ncells + 1)
    xx_st, yy_st = cp.meshgrid(x_stencil, x_stencil)
    distances = cp.sqrt(xx_st**2 + yy_st**2)
    stencil = distances < rad
    return stencil, xx_st, yy_st

def neighbors_gpu_distance_based(i, j, ii, jj, xx, yy, grid, cond_msk,
                                 radius, num_points, max_memory_gb=2.0):
    batch = cp.asarray([[int(i), int(j)]], dtype=cp.int32)
    neigh, counts = batch_neighbors_distance_based(
        batch, ii, jj, xx, yy, grid, cond_msk, radius, num_points, max_memory_gb
    )
    if counts[0] == 0:
        return cp.array([]).reshape(0, 5)
    return neigh[0, :counts[0], :]

def adaptive_radius_neighbors_gpu_safe(i, j, ii, jj, xx, yy, grid, cond_msk,
                                       initial_radius, min_points=8, max_points=32,
                                       max_radius=1e6, max_memory_gb=1.0):
    current_radius = initial_radius
    batch = cp.asarray([[int(i), int(j)]], dtype=cp.int32)
    while current_radius <= max_radius:
        neigh, counts = batch_neighbors_distance_based(
            batch, ii, jj, xx, yy, grid, cond_msk, current_radius, max_points, max_memory_gb
        )
        if counts[0] >= min_points:
            return neigh[0, :min(counts[0], max_points), :]
        current_radius *= 2.0
    return cp.array([]).reshape(0, 5)

def to_gpu(*arrays):
    return [cp.asarray(a) for a in arrays]

def to_cpu(*arrays):
    return [cp.asnumpy(a) for a in arrays]