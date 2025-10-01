#!/usr/bin/env bash
#SBATCH --array=0-19%2              # 20 grid configurations: 0..19
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/grid_config_homeostasis_%A_%a.out
#SBATCH --error=sbatch_err/grid_config_homeostasis_%A_%a.err
#SBATCH --job-name=run_lambdas_grid_configs_homeostasis

# ----------------------------
# Load environment
# ----------------------------
. ~/HomeostaticDANN/load_venv.sh

# ----------------------------
# Grid parameters
# ----------------------------
brightness_factors=(0 0.25 0.5 0.75)
lambda_homeos_values=(0.00001 0.0001 0.001 0.1 1)

# Fixed values
normtype=0
normtype_detach=1
excitatory_only=1
string_label="full"
HOMEOSTASIS=1
SHUNTING=1

# ----------------------------
# Sizes
# ----------------------------
num_brightness_factors=${#brightness_factors[@]}
num_lambda_homeos=${#lambda_homeos_values[@]}

# Optional sanity check
if (( num_brightness_factors * num_lambda_homeos != 20 )); then
    echo "WARNING: Grid size != 20; update --array accordingly." >&2
fi

# ----------------------------
# Index decode
# ----------------------------
grid_index=${SLURM_ARRAY_TASK_ID}

brightness_factor_idx=$(( grid_index % num_brightness_factors ))
lambda_homeos_idx=$(( (grid_index / num_brightness_factors) % num_lambda_homeos ))

brightness_factor=${brightness_factors[$brightness_factor_idx]}
LAMBDA_HOMEOS=${lambda_homeos_values[$lambda_homeos_idx]}

# ----------------------------
# Debug print
# ----------------------------
echo "GRID_INDEX:         $grid_index"
echo "brightness_factor:  $brightness_factor (idx $brightness_factor_idx)"
echo "LAMBDA_HOMEOS:      $LAMBDA_HOMEOS (idx $lambda_homeos_idx)"
echo "normtype_detach:    $normtype_detach"
echo "string_label:       $string_label"

# ----------------------------
# Export env vars for downstream script
# ----------------------------
export GRID_INDEX=$grid_index
export BRIGHTNESS_FACTOR=$brightness_factor
export NORMTYPE=$normtype
export HOMEOSTASIS=$HOMEOSTASIS
export LAMBDA_HOMEOS=$LAMBDA_HOMEOS
export NORMTYPE_DETACH=$normtype_detach
export SHUNTING=$SHUNTING
export EXCITATORY_ONLY=$excitatory_only
export LN_FEEDBACK="$string_label"

# ----------------------------
# Launch job
# ----------------------------
sbatch --export=ALL run_ei_homeostasis_network.sh
