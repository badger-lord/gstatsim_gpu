#!/bin/bash
#SBATCH --job-name=sgs_gpu_sim        # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=kasmith@nvidia.com  # Where to send mail	
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --cpus-per-task=8             # CPUs for data loading (adjust as needed)
#SBATCH --mem=64gb                    # Job memory request
#SBATCH --time=04:00:00               # Time limit hrs:min:sec
#SBATCH --output=logs/sgs_%j.log      # Standard output and error log
#SBATCH --partition=hpg-b200          # The B200 GPU partition on HPG, also HPG-turin for L4s
#SBATCH --gpus=b200:1                 # Request 1 B200 GPU
#SBATCH --account=vendor-nvidia     # Your UFRC account/group
#SBATCH --qos=vendor-nvidia          # Your QOS (usually same as group name)

# -----------------------------------------------------------------------------
# ENVIRONMENT SETUP
# -----------------------------------------------------------------------------

# 1. Load the Container Runtime
#module load singularity (ml = module load)
ml apptainer  #apptainer bought and owns singularity now. but names interchangable
# 2. Define Container Path
# UPDATE THIS PATH to where you ran the 'singularity pull' command to install continer
# CONTAINER_IMAGE="../rapids_25.06.sif" 
# UFRC has containers already uploaded on HPG as modules, we can use that
ml rapidsai/25.06

# 3. Define Project Paths
# Assuming this script is running from the 'scripts/' folder
# We need to bind the parent directory so the container can see 'gstatsim_custom'
PROJECT_ROOT=$(pwd)

# -----------------------------------------------------------------------------
# EXECUTION
# -----------------------------------------------------------------------------

echo "Starting SGS Simulation on $(hostname)"
echo "Using Container: $CONTAINER_IMAGE"
echo "Project Root: $PROJECT_ROOT"

# Explain the singularity flags: if using singularity
# --nv       : Passes the GPU drivers into the container (CRITICAL)
# --bind     : Mounts your project folder inside the container so Python can see your code

For our case
# python     : Runs the python command inside that container environment

python /blue/vendor-nvidia/smith.kaleb/gstatsim_custom/full_sim.py \
    --batch_size 16384 \
    --radius 50000 \
    --seed $SLURM_JOB_ID \
    --dtype float64 \
    --data_path "/blue/vendor-nvidia/smith.kaleb/gstatsim_custom/bedmap3_mod_1000.nc" \
    --vario_path "/blue/vendor-nvidia/smith.kaleb/gstatsim_custom/continental_variogram_1000.nc" \
    --output_dir "/blue/vendor-nvidia/smith.kaleb/gstatsim_custom/results" \
    --figure_dir "/blue/vendor-nvidia/smith.kaleb/gstatsim_custom/figures"

echo "Simulation Complete."
