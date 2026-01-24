# _krige_gpu.py

import cupy as cp
from tqdm import tqdm
from .covariance_gpu import batch_covariance_gpu

def make_rotation_matrix_gpu(azimuth, major_range, minor_range):
    # Convert to scalars if arrays are passed
    if hasattr(azimuth, 'item'):
        azimuth = float(azimuth.item())
    elif hasattr(azimuth, '__len__') and len(azimuth) == 1:
        azimuth = float(azimuth[0])
    else:
        azimuth = float(azimuth)
        
    if hasattr(major_range, 'item'):
        major_range = float(major_range.item())
    elif hasattr(major_range, '__len__') and len(major_range) == 1:
        major_range = float(major_range[0])
    else:
        major_range = float(major_range)
        
    if hasattr(minor_range, 'item'):
        minor_range = float(minor_range.item())
    elif hasattr(minor_range, '__len__') and len(minor_range) == 1:
        minor_range = float(minor_range[0])
    else:
        minor_range = float(minor_range)
    
    theta = (azimuth / 180.0) * cp.pi
    rot = cp.array([[cp.cos(theta), -cp.sin(theta)],
                    [cp.sin(theta),  cp.cos(theta)]], dtype=cp.float64)
    scale = cp.array([[1.0 / major_range, 0.0],
                      [0.0, 1.0 / minor_range]], dtype=cp.float64)
    return rot.dot(scale)


def pairwise_distances_batch(X):
    X2 = cp.sum(X * X, axis=2, keepdims=True)  # (B,N,1)
    Y2 = X2.transpose(0, 2, 1)                 # (B,1,N)
    inner = cp.matmul(X, X.transpose(0, 2, 1)) # (B,N,N)
    d2 = cp.maximum(X2 + Y2 - 2.0 * inner, 0.0)
    return cp.sqrt(d2)

def make_sigma_batch(coords_batch, rotation_matrix, vario):
    rotT = rotation_matrix.T
    mat = cp.tensordot(coords_batch, rotT, axes=([2], [0]))  # (B,N,2)
    norm_range = pairwise_distances_batch(mat)                # (B,N,N)
    vtype = vario.get('vtype', 'matern').lower()
    sill = vario.get('sill', 1.0)
    nugget = vario.get('nugget', 0.0)
    s = vario.get('s', 1.5)
    cov = batch_covariance_gpu(norm_range, vtype, sill, nugget,
                               batch_size=max(1_000_000, norm_range.size // 4), s=s)
    return cov

def make_rho_batch(coords_batch, sim_coords_batch, rotation_matrix, vario):
    rotT = rotation_matrix.T
    mat1 = cp.tensordot(coords_batch, rotT, axes=([2], [0]))  # (B,N,2)
    mat2 = cp.tensordot(sim_coords_batch, rotT, axes=([2], [0]))  # (B,N,2)
    diff = mat1 - mat2
    norm = cp.sqrt(cp.sum(diff * diff, axis=2))               # (B,N)
    vtype = vario.get('vtype', 'matern').lower()
    sill = vario.get('sill', 1.0)
    nugget = vario.get('nugget', 0.0)
    s = vario.get('s', 1.5)
    rho = batch_covariance_gpu(norm, vtype, sill, nugget,
                               batch_size=max(1_000_000, norm.size // 4), s=s)
    return rho

def _solve_group_ok(sim_points, neighbors_array, vario, normalize_weights=True):
    """
    Solve OK for a group with identical K (no NaN padding).
    neighbors_array: (B,K,5) with all rows fully populated (no NaNs).
    """
    B, K, _ = neighbors_array.shape
    coords = neighbors_array[:, :, 0:2]
    vals = neighbors_array[:, :, 2]

    rot = make_rotation_matrix_gpu(vario.get('azimuth', 0.0),
                                   vario.get('major_range', 1.0),
                                   vario.get('minor_range', 1.0))

    Sigma = make_sigma_batch(coords, rot, vario)  # (B,K,K)

    # Regularization (stronger)
    maxS = cp.max(cp.abs(Sigma), axis=(1, 2))
    eps = 1e-8 * (maxS + 1.0)                     # tuned up from 1e-10
    eyeK = cp.eye(K, dtype=Sigma.dtype)[None, :, :]
    Sigma = Sigma + eyeK * eps[:, None, None]

    # Build augmented A and RHS b
    A = cp.zeros((B, K + 1, K + 1), dtype=Sigma.dtype)
    A[:, :K, :K] = Sigma
    A[:, :K, K] = 1.0
    A[:, K, :K] = 1.0

    sim_rep = cp.repeat(sim_points[:, None, :], K, axis=1)
    rho = make_rho_batch(coords, sim_rep, rot, vario)  # (B,K)

    b = cp.zeros((B, K + 1), dtype=Sigma.dtype)
    b[:, :K] = rho
    b[:, K] = 1.0

    x = cp.linalg.solve(A, b[:, :, None])[:, :, 0]  # (B,K+1)
    w = x[:, :K]
    lam = x[:, K]  # unused

    if normalize_weights:
        wsum = cp.sum(w, axis=1, keepdims=True)
        w = cp.where(cp.abs(wsum) > 0, w / wsum, w)

    est = cp.sum(w * vals, axis=1)
    var = vario.get('sill', 1.0) - cp.sum(w * rho, axis=1)
    return est, var

def _solve_group_sk(sim_points, neighbors_array, vario, global_mean):
    B, K, _ = neighbors_array.shape
    coords = neighbors_array[:, :, 0:2]
    vals = neighbors_array[:, :, 2]

    rot = make_rotation_matrix_gpu(vario.get('azimuth', 0.0),
                                   vario.get('major_range', 1.0),
                                   vario.get('minor_range', 1.0))

    Sigma = make_sigma_batch(coords, rot, vario)  # (B,K,K)
    maxS = cp.max(cp.abs(Sigma), axis=(1, 2))
    eps = 1e-8 * (maxS + 1.0)
    eyeK = cp.eye(K, dtype=Sigma.dtype)[None, :, :]
    Sigma = Sigma + eyeK * eps[:, None, None]

    sim_rep = cp.repeat(sim_points[:, None, :], K, axis=1)
    rho = make_rho_batch(coords, sim_rep, rot, vario)  # (B,K)

    w = cp.linalg.solve(Sigma, rho[:, :, None])[:, :, 0]
    est = global_mean + cp.sum(w * (vals - global_mean), axis=1)
    var = vario.get('sill', 1.0) - cp.sum(w * rho, axis=1)
    return est, var

def _group_by_counts(neighbors_array, counts):
    """
    Split variable-length neighbor rows into groups with equal K.
    Returns list of (K, row_indices) where row_indices are GPU int32 arrays.
    """
    counts_cpu = cp.asnumpy(counts)
    groups = {}
    for idx, kcnt in enumerate(counts_cpu):
        if kcnt <= 0:
            continue
        groups.setdefault(int(kcnt), []).append(int(idx))
    groups_gpu = [(K, cp.asarray(rows, dtype=cp.int32)) for K, rows in groups.items()]
    return groups_gpu

# _krige_gpu.py - OPTIMIZED VERSION (same function signature)

def batch_ok_solve_gpu(sim_points, neighbors_array, vario, jitter_rel=3e-8):
    """
    Optimized version - same API, vectorized operations
    """
    import cupy as cp
    
    B, K, _ = neighbors_array.shape
    if K == 0 or B == 0:
        return cp.empty((0,), dtype=cp.float64), cp.empty((0,), dtype=cp.float64)

    coords = neighbors_array[:, :, 0:2]  # (B,K,2)
    vals = neighbors_array[:, :, 2]      # (B,K)
    valid = cp.isfinite(vals)            # (B,K)
    mfloat = valid.astype(cp.float64)

    # Replace NaNs in coords with zero; they'll be masked in covariances
    coords = cp.where(valid[..., None], coords, 0.0)

    # OPTIMIZATION: Single rotation matrix computation
    rot = make_rotation_matrix_gpu(vario.get('azimuth', 0.0),
                                 vario.get('major_range', 1.0),
                                 vario.get('minor_range', 1.0))

    # OPTIMIZATION: Vectorized rotation using single matrix multiply
    coords_r = coords @ rot.T  # (B,K,2) - single batched operation
    sim_r = sim_points @ rot.T  # (B,2)

    # OPTIMIZATION: More efficient pairwise distance computation
    coords_expanded = coords_r[:, :, None, :]     # (B,K,1,2)
    coords_transposed = coords_r[:, None, :, :]   # (B,1,K,2)
    diff = coords_expanded - coords_transposed     # (B,K,K,2)
    NR = cp.sqrt(cp.sum(diff**2, axis=3))         # (B,K,K)

    # OPTIMIZATION: Batch covariance with memory management
    vtype = vario.get('vtype', 'matern').lower()
    sill = float(vario.get('sill', 1.0))
    nugget = float(vario.get('nugget', 0.0))
    s = float(vario.get('s', 1.5))
    
    # Use larger batch size for better GPU utilization
    Sigma = batch_covariance_gpu(NR, vtype, sill, nugget, 
                               batch_size=max(2_000_000, NR.size // 2), s=s)
    
    # OPTIMIZATION: Vectorized masking
    valid_mask = mfloat[:, :, None] * mfloat[:, None, :]
    Sigma = Sigma * valid_mask

    # OPTIMIZATION: Vectorized regularization
    maxS = cp.max(cp.abs(Sigma), axis=(1,2), keepdims=True)
    eps = jitter_rel * (maxS + 1.0)
    eye_batch = cp.eye(K, dtype=Sigma.dtype)[None, :, :]
    Sigma = Sigma + eps * eye_batch

    # Build augmented matrix - same logic, optimized operations
    A = cp.zeros((B, K+1, K+1), dtype=Sigma.dtype)
    A[:, :K, :K] = Sigma
    A[:, :K, K] = mfloat
    A[:, K, :K] = mfloat

    # OPTIMIZATION: Vectorized RHS computation
    sim_expanded = sim_r[:, None, :]      # (B,1,2)
    diff_sim = coords_r - sim_expanded    # (B,K,2)
    nr = cp.sqrt(cp.sum(diff_sim**2, axis=2))  # (B,K)
    
    rho = batch_covariance_gpu(nr, vtype, sill, nugget, 
                             batch_size=max(1_000_000, nr.size), s=s)
    rho = rho * mfloat

    b = cp.zeros((B, K+1), dtype=Sigma.dtype)
    b[:, :K] = rho
    b[:, K] = 1.0

    # OPTIMIZATION: Batch linear solve with error handling
    try:
        x = cp.linalg.solve(A, b[:, :, None])[:, :, 0]  # (B,K+1)
    except cp.linalg.LinAlgError:
        # Fallback with stronger regularization
        A[:, :K, :K] += 1e-6 * eye_batch
        x = cp.linalg.solve(A, b[:, :, None])[:, :, 0]

    w = x[:, :K]

    # Normalize weights for numerical robustness - same logic
    wsum = cp.sum(w, axis=1, keepdims=True)
    w = cp.where(cp.abs(wsum) > 0, w / wsum, w)

    vals_safe = cp.where(valid, vals, 0.0)
    est = cp.sum(w * vals_safe, axis=1)
    var = sill - cp.sum(w * rho, axis=1)

    return est, var



def batch_sk_solve_gpu(sim_points, neighbors_array, vario, global_mean, batch_size=4096):
    B, Kpad, _ = neighbors_array.shape
    valid = ~cp.isnan(neighbors_array[:, :, 2])
    counts = cp.sum(valid, axis=1).astype(cp.int32)
    if not cp.any(counts > 0):
        return cp.zeros((0,), dtype=cp.float64), cp.zeros((0,), dtype=cp.float64)

    groups = _group_by_counts(neighbors_array, counts)
    all_est = cp.empty((neighbors_array.shape[0],), dtype=cp.float64)
    all_var = cp.empty_like(all_est)

    for K, rows in groups:
        neigh = neighbors_array[rows, :K, :]
        sims = sim_points[rows]
        est, var = _solve_group_sk(sims, neigh, vario, global_mean)
        all_est[rows] = est
        all_var[rows] = var

    return all_est, all_var

def to_gpu(*arrays):
    return [cp.asarray(a) for a in arrays]

def to_cpu(*arrays):
    return [cp.asnumpy(a) for a in arrays]
