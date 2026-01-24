# utilities_gpu.py

import cupy as cp
import numpy as np
from cuml.preprocessing import QuantileTransformer as GPU_QuantileTransformer
from copy import deepcopy
import skgstat as skg

def gaussian_transformation_gpu(grid, cond_msk, n_quantiles=500, cpu_fit=False, cpu_transformer=None, random_state=0):
    """
    Gaussian quantile transformation on GPU with exact CPU parity option.

    - If cpu_fit=False (default): fit and apply cuML QuantileTransformer on CPU copies
      of the conditioning data but keep outputs on GPU.
    - If cpu_fit=True: a scikit-learn QuantileTransformer instance must be passed
      via cpu_transformer; we apply its transform on CPU and send the result to GPU.
    """
    data_cond = grid[cond_msk].reshape(-1, 1)

    if cpu_fit:
        if cpu_transformer is None:
            raise ValueError("cpu_fit=True requires cpu_transformer (sklearn) to be provided.")
        data_cond_cpu = cp.asnumpy(data_cond)
        norm = cpu_transformer.transform(data_cond_cpu).squeeze()
        norm_gpu = cp.asarray(norm, dtype=cp.float64)
    else:
        # Fit cuML (GPU) transformer on CPU copy of the same data for determinism
        data_cond_cpu = cp.asnumpy(data_cond)
        nqt = GPU_QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="normal",
            random_state=random_state
        ).fit(data_cond_cpu)
        norm = nqt.transform(data_cond_cpu).squeeze()
        norm_gpu = cp.asarray(norm, dtype=cp.float64)

    grid_norm = cp.full(grid.shape, cp.nan, dtype=cp.float64)
    grid_norm[cond_msk] = norm_gpu

    # Return cuML transformer if we fitted it; else return the sklearn one passed in
    return grid_norm, (nqt if not cpu_fit else cpu_transformer)

def dists_to_cond_gpu(xx, yy, grid):
    cond_msk = ~cp.isnan(grid)
    x_cond = xx[cond_msk]
    y_cond = yy[cond_msk]
    xx_flat = xx.ravel()
    yy_flat = yy.ravel()
    n_points = xx_flat.size
    chunk = 20000
    min_dists = cp.full(n_points, cp.inf, dtype=cp.float64)
    for i in range(0, n_points, chunk):
        end = min(n_points, i + chunk)
        xs = xx_flat[i:end][:, None].astype(cp.float64)
        ys = yy_flat[i:end][:, None].astype(cp.float64)
        dx = xs - x_cond[None, :]
        dy = ys - y_cond[None, :]
        d = cp.sqrt(dx**2 + dy**2, dtype=cp.float64)
        min_dists[i:end] = cp.min(d, axis=1)
    return min_dists.reshape(xx.shape)

def dists_to_cond_gpu_chunked(xx, yy, grid, chunk_size=20000):
    return dists_to_cond_gpu(xx, yy, grid)

def get_random_generator_gpu(seed):
    if seed is None:
        rng = cp.random.default_rng()
    elif isinstance(seed, int):
        rng = cp.random.default_rng(seed=seed)
    else:
        rng = seed

    class RNGWrapper:
        def __init__(self, rng):
            self.rng = rng
        def normal(self, loc, scale, size):
            return loc + scale * self.rng.standard_normal(size)
        def standard_normal(self, size):
            return self.rng.standard_normal(size)

    return RNGWrapper(rng)

def variograms_gpu(xx, yy, grid, bin_func='even', maxlag=100e3, n_lags=70,
                   covmodels=['gaussian', 'spherical', 'exponential', 'matern'],
                   downsample=None, cpu_fit_transformer=None, use_cpu_transformer=False):
    """
    Compute experimental variogram using skgstat (CPU) while keeping arrays on GPU when possible.
    If use_cpu_transformer=True, apply the provided sklearn QuantileTransformer to ensure CPU/GPU parity.
    """
    cond_msk = ~cp.isnan(grid)
    grid_norm, nst_trans = gaussian_transformation_gpu(
        grid, cond_msk, cpu_fit=use_cpu_transformer, cpu_transformer=cpu_fit_transformer
    )

    x_cond = xx[cond_msk]
    y_cond = yy[cond_msk]
    data_norm = grid_norm[cond_msk]
    coords_cond = cp.stack([x_cond, y_cond], axis=1)

    if isinstance(downsample, int):
        coords_cond = coords_cond[::downsample]
        data_norm = data_norm[::downsample]

    coords_cpu = cp.asnumpy(coords_cond)
    data_cpu = cp.asnumpy(data_norm)

    V = skg.Variogram(coords_cpu, data_cpu, bin_func=bin_func, n_lags=n_lags, maxlag=maxlag, normalize=False)

    vgrams = {}
    for cov in covmodels:
        V_i = deepcopy(V)
        V_i.model = cov
        vgrams[cov] = V_i.parameters

    return vgrams, V.experimental, V.bins

def to_gpu(*arrays):
    return [cp.asarray(a) for a in arrays]

def to_cpu(*arrays):
    return [cp.asnumpy(a) for a in arrays]
