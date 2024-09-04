#!/usr/bin/env bash
#SBATCH --array=0-49  # 5 x 5 x 2 grid (50 configurations)
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
lambda_first_moment=(0.001 0.01 0.1 1 10)
lambda_second_moment=(0.001 0.01 0.1 1 10)
homeostasis_values=(1)  # Fixed to 1
normtypes=(0)  # Fixed to 0

# Calculate grid parameters based on SLURM_ARRAY_TASK_ID
num_brightness_factors=${#brightness_factors[@]}
num_lambda_first=${#lambda_first_moment[@]}
num_lambda_second=${#lambda_second_moment[@]}

grid_index=$SLURM_ARRAY_TASK_ID

brightness_factor_idx=$((grid_index % num_brightness_factors))
lambda_first_idx=$(( (grid_index / num_brightness_factors) % num_lambda_first ))
temp_index=$(( grid_index / (num_brightness_factors * num_lambda_first) ))
lambda_second_idx=$(( temp_index % num_lambda_second ))

brightness_factor=${brightness_factors[$brightness_factor_idx]}
lambda_first=${lambda_first_moment[$lambda_first_idx]}
lambda_second=${lambda_second_moment[$lambda_second_idx]}

# Export the selected parameters as environment variables
export GRID_INDEX=$grid_index
export BRIGHTNESS_FACTOR=$brightness_factor
export LAMBDA_FIRST=$lambda_first
export LAMBDA_SECOND=$lambda_second
export HOMEOSTASIS=1  # Fixed to 1
export NORMTYPE=0  # Fixed to 0

# Submit random jobs with the fixed set of random parameters
sbatch --export=ALL run_random_jobs_implicit_homeo.sh
