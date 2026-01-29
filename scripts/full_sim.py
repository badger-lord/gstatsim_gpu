"""
full_sim.py

Production script for GPU-Accelerated Sequential Gaussian Simulation (SGS).
Designed for execution on Slurm clusters with High-End GPUs (A100/H100/B200).

Usage:
    python full_sim.py --batch_size 16384 --radius 50000 --seed 123 --dtype float32
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import cupy as cp
import xarray as xr
import matplotlib.pyplot as plt

# Optional plotting library
try:
    from cmcrameri import cm 
except ImportError:
    cm = None

# Add local package to path (if running from folder)
sys.path.append('..')

try:
    # Try importing as a package first
    from gstatsim_custom import utilities_gpu, interpolate_gpu
except ImportError:
    # Fallback to local import if files are in the same directory
    try:
        import utilities_gpu
        import interpolate_gpu
    except ImportError:
        print("Error: Could not import helper modules (interpolate_gpu, utilities_gpu).")
        sys.exit(1)

# -----------------------------------------------------------------------------
# Configuration & Arguments
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run GPU SGS Simulation")
    
    # Core Parameters
    parser.add_argument("--batch_size", type=int, default=16384, help="GPU Batch Size (default: 16384)")
    parser.add_argument("--radius", type=float, default=50000.0, help="Search Radius in meters (default: 50km)")
    parser.add_argument("--num_points", type=int, default=32, help="Number of neighbors (default: 32)")
    parser.add_argument("--seed", type=int, default=0, help="Random Seed (default: 0)")
    
    # Precision Argument
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"],
                        help="Precision mode. float32 is faster and uses less VRAM.")

    # Paths
    parser.add_argument("--data_path", type=str, default="./bedmap3_mod_500.nc", help="Path to input NetCDF")
    parser.add_argument("--vario_path", type=str, default="./continental_variogram_1000.nc", help="Path to variogram NetCDF")
    parser.add_argument("--output_dir", type=str, default="../results", help="Directory to save results")
    parser.add_argument("--figure_dir", type=str, default="../figures", help="Directory to save plots")

    return parser.parse_args()

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    # 1. Setup Precision (Dtype)
    # -------------------------------------------------------------------------
    if args.dtype == "float32":
        dtype = cp.float32
        print("--- Mode: High Performance (float32) ---")
    else:
        dtype = cp.float64
        print("--- Mode: High Precision (float64) ---")

    print(f"Device: {cp.cuda.runtime.getDeviceCount()} GPU(s) detected")
    print(f"Batch Size: {args.batch_size}")
    print(f"Seed: {args.seed}")

    # Ensure output directories exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figure_dir).mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Data Loading & GPU Transfer
    # -------------------------------------------------------------------------
    print(f"Loading data from {args.data_path}...")
    try:
        ds = xr.open_dataset(Path(args.data_path))
    except FileNotFoundError:
        print(f"Error: Data file not found at {args.data_path}")
        sys.exit(1)

    print("Transferring data to GPU and casting to target dtype...")
    
    # Load and cast immediately to save memory
    mask_gpu = cp.asarray(ds.mask.values) # Keep int/bool
    thick_cond_cpu = ds.thick_cond.values
    surface_topo_gpu = cp.asarray(ds.surface_topography.values, dtype=dtype)
    
    # Condition: Bedrock = Surface - Thickness (where mask allows)
    # Mask 4 = Rock Outcrop (Thickness 0)
    # We use explicit casting for thickness to match the surface_topo dtype
    thick_cond_gpu = cp.where(mask_gpu == 4, 0.0, cp.asarray(thick_cond_cpu, dtype=dtype))
    
    bed_cond_gpu = surface_topo_gpu - thick_cond_gpu

    # Define Simulation Mask (1=Floating Ice, 2=Grounded Ice, 4=Rock)
    ice_rock_msk_gpu = (mask_gpu == 1) | (mask_gpu == 4) | (mask_gpu == 2)
    
    # Filter conditioning data to valid areas
    bed_cond_gpu = cp.where(ice_rock_msk_gpu, bed_cond_gpu, cp.nan)

    # Coordinate Grids
    xx_gpu, yy_gpu = cp.meshgrid(
        cp.asarray(ds.x.values, dtype=dtype),
        cp.asarray(ds.y.values, dtype=dtype),
        indexing='ij'
    )

    # -------------------------------------------------------------------------
    # 3. Pre-processing (Trends & Variograms)
    # -------------------------------------------------------------------------
    print("Pre-processing: De-trending and Normal Score Transform...")
    
    # Trend Removal
    trend_gpu = cp.asarray(ds.trend.values, dtype=dtype)
    res_cond_gpu = bed_cond_gpu - trend_gpu

    # Variogram Parameters
    print(f"Loading variogram from {args.vario_path}...")
    dsv = xr.load_dataset(Path(args.vario_path))

    def extract_scalar(arr):
        if hasattr(arr, 'size') and arr.size > 1: return float(np.nanmedian(arr))
        elif hasattr(arr, 'item'): return float(arr.item())
        return float(arr)

    vario_params = {
        'azimuth': extract_scalar(dsv.azimuth.values),
        'nugget': 0,
        'major_range': extract_scalar(dsv.major_range.values),
        'minor_range': extract_scalar(dsv.minor_range.values),
        'sill': extract_scalar(dsv.sill.values),
        's': extract_scalar(dsv.smooth.values),
        'vtype': 'matern',
    }

    # -------------------------------------------------------------------------
    # 4. SGS Simulation
    # -------------------------------------------------------------------------
    print(f"Starting Simulation Loop...")
    tic = time.time()

    try:
        sim_gpu = interpolate_gpu.sgs_gpu(
            xx_gpu, yy_gpu,
            res_cond_gpu,
            vario_params,
            radius=args.radius,
            num_points=args.num_points,
            seed=args.seed,
            batch_size=args.batch_size,
            quiet=False,
            sim_mask=ice_rock_msk_gpu,
            max_memory_gb=150.0,
            dtype=dtype  # 
        )

    except Exception as e:
        print(f"CRITICAL ERROR during simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    toc = time.time()
    runtime = toc - tic
    print(f"Simulation completed in {runtime:.2f} seconds")

    # -------------------------------------------------------------------------
    # 5. Saving Results
    # -------------------------------------------------------------------------
    # Include dtype in filename for clarity
    output_path = Path(args.output_dir) / f"sim_seed{args.seed}_batch{args.batch_size}_{args.dtype}.npy"
    print(f"Saving raw output to {output_path}...")

    # Fetch to CPU
    sim_cpu = cp.asnumpy(sim_gpu)
    np.save(output_path, sim_cpu)

    # Save timing metric
    time_path = Path(args.output_dir) / f"timing_seed{args.seed}_{args.dtype}.txt"
    with open(time_path, 'w') as f:
        f.write(f"{runtime:.4f}")

    # -------------------------------------------------------------------------
    # 6. Plotting (Headless)
    # -------------------------------------------------------------------------
    print("Generating validation plot...")
    
    # Add trend back
    trend_cpu = cp.asnumpy(trend_gpu)
    final_elevation = sim_cpu + trend_cpu

    # Coordinates for plotting
    xx = cp.asnumpy(xx_gpu)
    yy = cp.asnumpy(yy_gpu)

    # Create Figure (Headless)
    plt.figure(figsize=(12, 10))
    
    # Use batlowW if available, else viridis
    cmap = 'viridis'
    if cm is not None and hasattr(cm, 'batlowW'):
        cmap = cm.batlowW

    # Subsample for faster plotting if grid is huge
    step = 2
    im = plt.pcolormesh(
        xx[::step, ::step]/1000,
        yy[::step, ::step]/1000,
        final_elevation[::step, ::step],
        cmap=cmap,
        shading='auto'
    )

    plt.axis('scaled')
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
    plt.title(f'SGS Simulation (Seed {args.seed})\nRadius: {args.radius/1000}km, Neighbors: {args.num_points}, Dtype: {args.dtype}')
    plt.colorbar(im, label='Bed Elevation [m]', pad=0.02)

    plot_path = Path(args.figure_dir) / f"sim_plot_seed{args.seed}_{args.dtype}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close() # Free memory
    
    print(f"Plot saved to {plot_path}")
    print("Done.")

if __name__ == "__main__":
    main()
