#!/usr/bin/env bash
#SBATCH --array=0-1  # Adjust based on the number of grid configurations
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/grid_config_homeostasis_%A_%a.out
#SBATCH --error=sbatch_err/grid_config_homeostasis_%A_%a.err
#SBATCH --job-name=run_grid_configs_homeostasis

# Load environment
. /etc/profile
module load anaconda/3
conda activate ffcv_eg

# Grid parameters
brightness_factors=(0 0.75)
lambda_homeos=(300)  # Included in this script
homeostasis_values=(1)
normtypes=(0)  # Fixed to 0

# Calculate grid parameters based on SLURM_ARRAY_TASK_ID
num_brightness_factors=${#brightness_factors[@]}
num_lambda_homeos=${#lambda_homeos[@]}
num_homeostasis=${#homeostasis_values[@]}
num_normtypes=${#normtypes[@]}

grid_index=$SLURM_ARRAY_TASK_ID


brightness_factor=${brightness_factors[$grid_index]}

# Load the pre-generated random configurations
export GRID_INDEX=$grid_index
export BRIGHTNESS_FACTOR=$brightness_factor
export LAMBDA_HOMEOS=300
export HOMEOSTASIS=1  # Fixed to 1
export NORMTYPE=0  # Fixed to 0

# Submit random jobs with the fixed set of random parameters
sbatch --export=ALL run_random_jobs.sh
