# covariance_gpu.py

import cupy as cp

# Use SciPy on CPU only if you fall back; not needed when using kv_gpu
# from scipy.special import kv as scipy_kv, gamma as scipy_gamma

from cupyx.scipy.special import gamma as cupy_gamma
from .besselk_gpu import kv_gpu  # ensure this file exists in the same package

def _ensure_array(x):
    return x if isinstance(x, cp.ndarray) else cp.asarray(x)

def exponential_cov_norm_gpu(norm_range, sill, nugget, **kwargs):
    nr = _ensure_array(norm_range).astype(cp.float64, copy=False)
    return (float(sill) - float(nugget)) * cp.exp(-3.0 * nr)

def gaussian_cov_norm_gpu(norm_range, sill, nugget, **kwargs):
    nr = _ensure_array(norm_range).astype(cp.float64, copy=False)
    return (float(sill) - float(nugget)) * cp.exp(-3.0 * cp.square(nr))

def spherical_cov_norm_gpu(norm_range, sill, nugget, **kwargs):
    nr = _ensure_array(norm_range).astype(cp.float64, copy=False)
    c = float(sill) - float(nugget) - 1.5 * nr + 0.5 * cp.power(nr, 3)
    c = cp.where(nr > 1.0, float(sill) - 1.0, c)
    return c

def matern_cov_norm_gpu(norm_range, sill, nugget, s, **kwargs):
    nr = cp.asarray(norm_range, dtype=cp.float64)
    s = float(s)
    sill = float(sill); nugget = float(nugget)
    r = cp.where(nr == 0.0, 1e-8, nr)
    scale = 0.45246434 * cp.exp(-0.70449189 * s) + 1.7863836
    z = scale * r * cp.sqrt(s)
    kv_vals = kv_gpu(s, 2.0 * z, scaled=False)
    coeff = (sill - nugget) * 2.0 / cupy_gamma(s)  # gamma(s) on device
    c = coeff * cp.power(z, s) * kv_vals
    c = cp.where(cp.isnan(c), (sill - nugget), c)
    return c

covmodels_gpu = {
    'matern': matern_cov_norm_gpu,
    'exponential': exponential_cov_norm_gpu,
    'gaussian': gaussian_cov_norm_gpu,
    'spherical': spherical_cov_norm_gpu,
}


def batch_covariance_gpu(distances, model_type, sill, nugget, batch_size=1_000_000, **kwargs):
    """
    Optimized version - same API, better memory management
    """
    distances = _ensure_array(distances).astype(cp.float64, copy=False)
    
    if model_type not in covmodels_gpu:
        raise ValueError(f"Unknown model type {model_type}")
    
    func = covmodels_gpu[model_type]
    
    # OPTIMIZATION: Adaptive batch sizing based on GPU memory
    if distances.size <= batch_size:
        return func(distances, sill, nugget, **kwargs)
    
    # OPTIMIZATION: Better memory management for large arrays
    flat = distances.ravel()
    out = cp.empty_like(flat, dtype=cp.float64)
    
    # OPTIMIZATION: Use larger batch sizes when possible
    mempool = cp.get_default_memory_pool()
    available_memory = mempool.free_bytes() + mempool.used_bytes() - mempool.used_bytes()
    
    # Adaptive batch size based on available memory
    memory_per_element = 16  # Conservative estimate
    adaptive_batch_size = min(batch_size, max(100000, int(available_memory * 0.5 / memory_per_element)))
    
    for i in range(0, flat.size, adaptive_batch_size):
        j = min(i + adaptive_batch_size, flat.size)
        chunk = flat[i:j]
        out[i:j] = func(chunk, sill, nugget, **kwargs)
    
    return out.reshape(distances.shape)

