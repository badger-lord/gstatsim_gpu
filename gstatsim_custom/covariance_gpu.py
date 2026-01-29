"""
covariance_gpu.py

This module defines standard covariance functions for geostatistics.
All functions operate on CuPy arrays.

Supported Models:
- Exponential
- Gaussian
- Spherical
- Matern (requires besselk_gpu)
"""

import cupy as cp
from cupyx.scipy.special import gamma as cupy_gamma
from .besselk_gpu import kv_gpu # Imports the custom CUDA Bessel kernel

def _ensure_array(x):
    """Helper to ensure input is a CuPy array."""
    return x if isinstance(x, cp.ndarray) else cp.asarray(x)

def exponential_cov_norm_gpu(norm_range, sill, nugget, **kwargs):
    """Exponential covariance model."""
    nr = _ensure_array(norm_range).astype(cp.float64, copy=False)
    return (float(sill) - float(nugget)) * cp.exp(-3.0 * nr)

def gaussian_cov_norm_gpu(norm_range, sill, nugget, **kwargs):
    """Gaussian covariance model."""
    nr = _ensure_array(norm_range).astype(cp.float64, copy=False)
    return (float(sill) - float(nugget)) * cp.exp(-3.0 * cp.square(nr))

def spherical_cov_norm_gpu(norm_range, sill, nugget, **kwargs):
    """Spherical covariance model."""
    nr = _ensure_array(norm_range).astype(cp.float64, copy=False)
    c = float(sill) - float(nugget) - 1.5 * nr + 0.5 * cp.power(nr, 3)
    c = cp.where(nr > 1.0, float(sill) - 1.0, c)
    return c

def matern_cov_norm_gpu(norm_range, sill, nugget, s, **kwargs):
    """
    Matern covariance model.
    
    Parameters:
    -----------
    s : float
        The smoothness parameter (often denoted as nu or alpha).
    """
    nr = cp.asarray(norm_range, dtype=cp.float64)
    s = float(s)
    sill = float(sill)
    nugget = float(nugget)
    
    # Avoid division by zero at lag 0
    r = cp.where(nr == 0.0, 1e-8, nr)
    
    # Empirical scaling factors (standard GSLIB/gstat conventions)
    scale = 0.45246434 * cp.exp(-0.70449189 * s) + 1.7863836
    z = scale * r * cp.sqrt(s)
    
    # Compute Bessel K using custom CUDA kernel
    kv_vals = kv_gpu(s, 2.0 * z, scaled=False)
    
    coeff = (sill - nugget) * 2.0 / cupy_gamma(s) 
    c = coeff * cp.power(z, s) * kv_vals
    
    # Handle the singularity at distance 0 (where correlation is 1.0 * sill)
    c = cp.where(cp.isnan(c), (sill - nugget), c)
    return c

# Registry of available models
covmodels_gpu = {
    'matern': matern_cov_norm_gpu,
    'exponential': exponential_cov_norm_gpu,
    'gaussian': gaussian_cov_norm_gpu,
    'spherical': spherical_cov_norm_gpu,
}

def batch_covariance_gpu(distances, model_type, sill, nugget, batch_size=1_000_000, **kwargs):
    """
    Compute covariance for a large array of distances by splitting into batches.
    This prevents Out-Of-Memory (OOM) errors on the GPU.
    
    Parameters:
    -----------
    distances : cp.ndarray
        Array of lag distances (any shape).
    model_type : str
        Name of the model ('matern', 'spherical', etc.).
    batch_size : int
        Target number of elements to process at once.
    """
    distances = _ensure_array(distances).astype(cp.float64, copy=False)
    
    if model_type not in covmodels_gpu:
        raise ValueError(f"Unknown model type {model_type}")
        
    func = covmodels_gpu[model_type]
    
    # If small enough, run directly
    if distances.size <= batch_size:
        return func(distances, sill, nugget, **kwargs)
        
    # Flatten for chunked processing
    flat = distances.ravel()
    out = cp.empty_like(flat, dtype=cp.float64)
    
    # OPTIMIZATION: Adaptive batch sizing based on available GPU memory
    mempool = cp.get_default_memory_pool()
    available_memory = mempool.free_bytes() + mempool.used_bytes() - mempool.used_bytes()
    
    memory_per_element = 16 # Conservative estimate (float64 inputs + outputs)
    adaptive_batch_size = min(batch_size, max(100000, int(available_memory * 0.5 / memory_per_element)))
    
    # Process in chunks
    for i in range(0, flat.size, adaptive_batch_size):
        j = min(i + adaptive_batch_size, flat.size)
        chunk = flat[i:j]
        out[i:j] = func(chunk, sill, nugget, **kwargs)
        
    return out.reshape(distances.shape)
