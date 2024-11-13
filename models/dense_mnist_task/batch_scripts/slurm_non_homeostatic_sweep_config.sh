#!/usr/bin/env bash
#SBATCH --array=0-3  # 4 grid configurations: 0, 1, 2, 3
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/grid_config_no_homeostasis_%A_%a.out
#SBATCH --error=sbatch_err/grid_config_no_homeostasis_%A_%a.err
#SBATCH --job-name=run_grid_configs_no_homeostasis

# Load environment
. /etc/profile
module load anaconda/3
conda activate ffcv_eg

# Grid parameters
brightness_factors=(0.25 0.75)
homeostasis_values=(0)
normtypes=(0 1)
export SHUNTING=1

# Calculate grid parameters based on SLURM_ARRAY_TASK_ID
num_brightness_factors=${#brightness_factors[@]}
num_normtypes=${#normtypes[@]}

grid_index=$SLURM_ARRAY_TASK_ID

brightness_factor_idx=$((grid_index % num_brightness_factors))
normtype_idx=$((grid_index / num_brightness_factors))

brightness_factor=${brightness_factors[$brightness_factor_idx]}
normtype=${normtypes[$normtype_idx]}

# Load the pre-generated random configurations
export GRID_INDEX=$grid_index
export BRIGHTNESS_FACTOR=$brightness_factor
export NORMTYPE=$normtype
export NORMTYPE_DETACH=1
export HOMEOSTASIS=0  # Fixed to 0
export LAMBDA_HOMEOS=1 # Fixed to 1 But not functional because homeostasis is deactivated

# Submit random jobs with the fixed set of random parameters
sbatch --export=ALL run_random_jobs.sh
