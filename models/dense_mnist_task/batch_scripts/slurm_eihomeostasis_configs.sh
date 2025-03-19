#!/usr/bin/env bash
#SBATCH --array=0-7  # 8 grid configurations: 0 to 7
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
brightness_factors=(0 0.5 0.75 1)
homeostasis_values=(0)  # Fixed to 0
normtypes=(0)           # Fixed to 0
normtype_detach=(0 1)
excitatory_only=(1)     # Fixed to 1

# Calculate grid parameters based on SLURM_ARRAY_TASK_ID
num_brightness_factors=${#brightness_factors[@]}
num_normtypes=${#normtypes[@]}
num_normtype_detach=${#normtype_detach[@]}
num_excitatory_only=${#excitatory_only[@]}

grid_index=$SLURM_ARRAY_TASK_ID

brightness_factor_idx=$((grid_index % num_brightness_factors))
detach_normtype_idx=$((grid_index / num_brightness_factors % num_normtype_detach))

brightness_factor=${brightness_factors[$brightness_factor_idx]}
normtype=${normtypes[0]}  # Always 0
detach_normtype=${normtype_detach[$detach_normtype_idx]}
excitatory_only=${excitatory_only[0]}  # Always 1

# Load the pre-generated random configurations
export GRID_INDEX=$grid_index
export BRIGHTNESS_FACTOR=$brightness_factor
export NORMTYPE=$normtype
export HOMEOSTASIS=1
export LAMBDA_HOMEOS=0.001
export NORMTYPE_DETACH=$detach_normtype
export SHUNTING=0
export EXCITATORY_ONLY=$excitatory_only

# Submit random jobs with the fixed set of random parameters
sbatch --export=ALL run_eihomeostasis_configs.sh
