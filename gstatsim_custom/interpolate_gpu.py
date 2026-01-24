# interpolate_gpu.py

import cupy as cp
from copy import deepcopy
import numbers
from tqdm import tqdm
from .utilities_gpu import gaussian_transformation_gpu, get_random_generator_gpu, variograms_gpu
from ._krige_gpu import batch_ok_solve_gpu, batch_sk_solve_gpu
from .neighbors_gpu import batch_neighbors_distance_based, make_circle_stencil_gpu_safe

def _sanity_checks_gpu(xx, yy, grid, vario, radius, num_points, ktype):
    if not isinstance(xx, cp.ndarray) or xx.ndim != 2:
        raise ValueError('xx must be a 2D CuPy array')
    if not isinstance(yy, cp.ndarray) or yy.ndim != 2:
        raise ValueError('yy must be a 2D CuPy array')
    if not isinstance(grid, cp.ndarray) or grid.ndim != 2:
        raise ValueError('grid must be a 2D CuPy array')
    expected = ['major_range', 'minor_range', 'azimuth','sill','nugget','vtype']
    for k in expected:
        if k not in vario:
            raise ValueError(f"Missing variogram key {k}")
    if vario['vtype'].lower() == 'matern' and 's' not in vario:
        raise ValueError("Matern requires 's' parameter")

def _preprocess_gpu_safe(xx, yy, grid, variogram, sim_mask, radius, stencil, max_memory_gb):
    cond_msk = ~cp.isnan(grid)
    out_grid, nst_trans = gaussian_transformation_gpu(grid, cond_msk)
    if sim_mask is None:
        sim_mask = cp.full(xx.shape, True)
    ii, jj = cp.meshgrid(cp.arange(xx.shape[0]), cp.arange(xx.shape[1]), indexing='ij')
    inds = cp.stack([ii[sim_mask].flatten(), jj[sim_mask].flatten()], axis=1)
    vario = deepcopy(variogram)
    for k in vario:
        if isinstance(vario[k], numbers.Number):
            vario[k] = float(vario[k])
    global_mean = float(cp.mean(out_grid[cond_msk]))
    if stencil is None:
        stencil_res = make_circle_stencil_gpu_safe(xx[0, :], radius, max_memory_gb)
        stencil = stencil_res if stencil_res[0] is not None else None
    return out_grid, nst_trans, cond_msk, inds, vario, global_mean, stencil

def krige_gpu(xx, yy, grid, variogram, radius=100e3, num_points=32, ktype='ok',
              sim_mask=None, quiet=False, stencil=None, max_memory_gb=20.0, batch_size=None,
              use_sector_balance=True, n_sectors=8):
    _sanity_checks_gpu(xx, yy, grid, variogram, radius, num_points, ktype)
    out_grid, nst_trans, cond_msk, inds, vario, global_mean, stencil = _preprocess_gpu_safe(
        xx, yy, grid, variogram, sim_mask, radius, stencil, max_memory_gb
    )
    ii, jj = cp.meshgrid(cp.arange(xx.shape[0]), cp.arange(xx.shape[1]), indexing='ij')

    if batch_size is None:
        bytes_per_point = num_points * 5 * 8
        batch_size = int((max_memory_gb * (1024**3)) // bytes_per_point)
        batch_size = max(2048, min(batch_size, inds.shape[0]))

    n_points = inds.shape[0]

    # Progress bar
    pbar = tqdm(total=n_points, desc="Kriging", unit="pts")

    for bstart in range(0, n_points, batch_size):
        bend = min(n_points, bstart + batch_size)
        batch_inds = inds[bstart:bend].astype(cp.int32)

        # Build neighbors for the batch in one GPU call
        neighbors, counts = batch_neighbors_distance_based(batch_inds, ii, jj, xx, yy, out_grid, cond_msk, radius, num_points, max_memory_gb)

        # Sim points coords
        sim_points = cp.stack([xx[batch_inds[:,0], batch_inds[:,1]], yy[batch_inds[:,0], batch_inds[:,1]]], axis=1)

        # Remove rows with zero neighbors by setting defaults
        valid_mask = counts > 0

        if not cp.any(valid_mask):
            pbar.update(bend - bstart)
            continue

        sim_points_valid = sim_points[valid_mask]
        neighbors_valid = neighbors[valid_mask]

        if ktype == 'ok':
            ests, vars_ = batch_ok_solve_gpu(sim_points_valid, neighbors_valid, vario)
        else:
            ests, vars_ = batch_sk_solve_gpu(sim_points_valid, neighbors_valid, vario, global_mean)

        # Fill into out_grid (only the valid indices)
        # Map back to original batch positions
        valid_indices = cp.where(valid_mask)[0]
        for idx_local, idx_valid in enumerate(valid_indices):
            i, j = int(batch_inds[idx_valid,0]), int(batch_inds[idx_valid,1])
            out_grid[i, j] = ests[idx_local]

        pbar.update(bend - bstart)

    pbar.close()

    # Back-transform to original distribution
    sim_trans = nst_trans.inverse_transform(out_grid.reshape(-1, 1)).squeeze().reshape(xx.shape)
    return sim_trans


def sgs_gpu(xx, yy, grid, variogram, radius=100e3, num_points=32, ktype='ok',
            sim_mask=None, quiet=False, stencil=None, seed=None, max_memory_gb=23.0,
            batch_size=None, use_sector_balance=True, n_sectors=8):
    _sanity_checks_gpu(xx, yy, grid, variogram, radius, num_points, ktype)
    out_grid, nst_trans, cond_msk, inds, vario, global_mean, stencil = _preprocess_gpu_safe(
        xx, yy, grid, variogram, sim_mask, radius, stencil, max_memory_gb
    )

    rng = cp.random.default_rng(seed)
    shuffled = cp.copy(inds)
    cp.random.shuffle(shuffled)

    ii, jj = cp.meshgrid(cp.arange(xx.shape[0]), cp.arange(xx.shape[1]), indexing='ij')
    if batch_size is None:
        bytes_per_point = num_points * 5 * 8
        batch_size = int((max_memory_gb * (1024**3)) // bytes_per_point)
        batch_size = max(2048, min(batch_size, shuffled.shape[0]))

    n_points = shuffled.shape[0]

    # Progress bar
    pbar = tqdm(total=n_points, desc="SGS", unit="pts")

    for bstart in range(0, n_points, batch_size):
        bend = min(n_points, bstart + batch_size)
        batch_inds = shuffled[bstart:bend].astype(cp.int32)

        neighbors, counts = batch_neighbors_distance_based(batch_inds, ii, jj, xx, yy, out_grid, cond_msk, radius, num_points, max_memory_gb)

        sim_points = cp.stack([xx[batch_inds[:,0], batch_inds[:,1]], yy[batch_inds[:,0], batch_inds[:,1]]], axis=1)

        valid_mask = counts > 0

        if not cp.any(valid_mask):
            pbar.update(bend - bstart)
            continue

        sim_points_valid = sim_points[valid_mask]
        neighbors_valid = neighbors[valid_mask]

        if ktype == 'ok':
            ests, vars_ = batch_ok_solve_gpu(sim_points_valid, neighbors_valid, vario)
        else:
            ests, vars_ = batch_sk_solve_gpu(sim_points_valid, neighbors_valid, vario, global_mean)


        # sample from normals in batch: vectorized
        vars_safe = cp.abs(vars_)
        samp = rng.standard_normal(size=ests.shape) * cp.sqrt(vars_safe) + ests

        # write back into out_grid
        valid_indices = cp.where(valid_mask)[0]
        for idx_local, idx_valid in enumerate(valid_indices):
            i, j = int(batch_inds[idx_valid,0]), int(batch_inds[idx_valid,1])
            out_grid[i, j] = samp[idx_local]
            cond_msk[i, j] = True

        pbar.update(bend - bstart)

    pbar.close()

    sim_trans = nst_trans.inverse_transform(out_grid.reshape(-1, 1)).squeeze().reshape(xx.shape)
    return sim_trans
