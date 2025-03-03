#!/usr/bin/env bash
#SBATCH --array=0-3  # Adjusted for the added homeostasis_gradient variable
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/grid_config_homeostasis_%A_%a.out
#SBATCH --error=sbatch_err/grid_config_homeostasis_%A_%a.err
#SBATCH --job-name=run_grid_configs_homeostasis

# Load environment
. ~/HomeostaticDANN/load_venv.sh

# Grid parameters
brightness_factors=(0 1)
homeostasis_values=(1)
normtypes=(0)  # Fixed to 0
homeostasis_gradients=(0 1)  # New variable with options 0 and 1

# Calculate grid dimensions
num_brightness_factors=${#brightness_factors[@]}
num_homeostasis=${#homeostasis_values[@]}
num_normtypes=${#normtypes[@]}
num_homeostasis_gradients=${#homeostasis_gradients[@]}

total_configs=$((num_brightness_factors * num_homeostasis_gradients))
grid_index=$SLURM_ARRAY_TASK_ID

# Determine corresponding parameters
brightness_index=$((grid_index % num_brightness_factors))
homeostasis_gradient_index=$((grid_index / num_brightness_factors))

brightness_factor=${brightness_factors[$brightness_index]}
homeostasis_gradient=${homeostasis_gradients[$homeostasis_gradient_index]}

# Export parameters
export GRID_INDEX=$grid_index
export BRIGHTNESS_FACTOR=$brightness_factor
export HOMEOSTASIS=1  # Fixed to 1
export NORMTYPE=0  # Fixed to 0
export HOMEOSTASIS_GRADIENT=$homeostasis_gradient
export LAMBDA_HOMEOS=0.1 # THIS IS WHAT NEEDS TO BE CHANGED IN THE SCRIPT
export SHUNTING=0
export NORMTYPE_DETACH=0

# Submit random jobs with the fixed set of random parameters
sbatch --export=ALL run_random_jobs.sh
