#!/usr/bin/env bash
#SBATCH --array=0-31%2              # 32 grid configurations: 0..31
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/grid_config_homeostasis_%A_%a.out
#SBATCH --error=sbatch_err/grid_config_homeostasis_%A_%a.err
#SBATCH --job-name=run_grid_configs_homeostasis

# ----------------------------
# Load environment
# ----------------------------
. ~/HomeostaticDANN/load_venv.sh

# ----------------------------
# Grid parameters
# ----------------------------
brightness_factors=(0 0.25 0.5 0.75)
homeostasis_values=(0)        # Fixed to 0
normtypes=(0)                 # Fixed to 0
normtype_detach=(0 1)
excitatory_only=(1)           # Fixed to 1

# >>> Updated list of 4 strings <<<
string_labels=("scale" "center" "decorrelate" "full")

# ----------------------------
# Sizes
# ----------------------------
num_brightness_factors=${#brightness_factors[@]}
num_normtype_detach=${#normtype_detach[@]}
num_string_labels=${#string_labels[@]}

# Optional sanity check
if (( num_brightness_factors * num_normtype_detach * num_string_labels != 32 )); then
    echo "WARNING: Grid size != 32; update --array accordingly." >&2
fi

# ----------------------------
# Index decode
# ----------------------------
grid_index=${SLURM_ARRAY_TASK_ID}

brightness_factor_idx=$(( grid_index % num_brightness_factors ))
detach_normtype_idx=$(( (grid_index / num_brightness_factors) % num_normtype_detach ))
string_label_idx=$(( (grid_index / (num_brightness_factors * num_normtype_detach)) % num_string_labels ))

brightness_factor=${brightness_factors[$brightness_factor_idx]}
detach_normtype=${normtype_detach[$detach_normtype_idx]}
string_label=${string_labels[$string_label_idx]}

# Fixed values
normtype=${normtypes[0]}
excitatory_only=${excitatory_only[0]}
HOMEOSTASIS=1
LAMBDA_HOMEOS=0.01
SHUNTING=1

# ----------------------------
# Debug print
# ----------------------------
echo "GRID_INDEX:         $grid_index"
echo "brightness_factor:  $brightness_factor (idx $brightness_factor_idx)"
echo "detach_normtype:    $detach_normtype (idx $detach_normtype_idx)"
echo "string_label:       $string_label (idx $string_label_idx)"

# ----------------------------
# Export env vars for downstream script
# ----------------------------
export GRID_INDEX=$grid_index
export BRIGHTNESS_FACTOR=$brightness_factor
export NORMTYPE=$normtype
export HOMEOSTASIS=$HOMEOSTASIS
export LAMBDA_HOMEOS=$LAMBDA_HOMEOS
export NORMTYPE_DETACH=$detach_normtype
export SHUNTING=$SHUNTING
export EXCITATORY_ONLY=$excitatory_only
export LN_FEEDBACK="$string_label"

# ----------------------------
# Launch job
# ----------------------------
sbatch --export=ALL run_ei_homeostasis_network.sh
