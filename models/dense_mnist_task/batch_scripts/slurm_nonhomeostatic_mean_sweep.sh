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
. ~/HomeostaticDANN/load_venv.sh

# Grid parameters
brightness_factors=(0 0.5 0.75 1)
homeostasis_values=(0)
normtype=1
normtype_detach=1  # Fixed to 1

# Calculate grid parameters based on SLURM_ARRAY_TASK_ID
grid_index=$SLURM_ARRAY_TASK_ID
brightness_factor=${brightness_factors[$grid_index]}

# Load the pre-generated random configurations
export GRID_INDEX=$grid_index
export BRIGHTNESS_FACTOR=$brightness_factor
export NORMTYPE=$normtype
export HOMEOSTASIS=0  # Fixed to 0
export LAMBDA_HOMEOS=1 # Fixed to 1 But not functional because homeostasis is deactivated
export NORMTYPE_DETACH=$normtype_detach
export SHUNTING=0
export HOMEOSTASIS_GRADIENT=0

# Submit random jobs with the fixed set of random parameters
sbatch --export=ALL run_random_jobs.sh





# #!/usr/bin/env bash
# #SBATCH --array=0-7  # 8 grid configurations: 0, 1, 2, 3, 4, 5, 6, 7
# #SBATCH --partition=long
# #SBATCH --gres=gpu:rtx8000:1
# #SBATCH --mem=16GB
# #SBATCH --time=24:00:00
# #SBATCH --cpus-per-gpu=4
# #SBATCH --output=sbatch_out/grid_config_no_homeostasis_%A_%a.out
# #SBATCH --error=sbatch_err/grid_config_no_homeostasis_%A_%a.err
# #SBATCH --job-name=run_grid_configs_no_homeostasis

# # Load environment
# . ~/HomeostaticDANN/load_venv.sh

# # Grid parameters
# brightness_factors=(0 0.5 0.75 1)
# homeostasis_values=(0)
# normtypes=(0 1)
# normtype_detach=(0 1)

# # Calculate grid parameters based on SLURM_ARRAY_TASK_ID
# num_brightness_factors=${#brightness_factors[@]}
# num_normtypes=${#normtypes[@]}

# grid_index=$SLURM_ARRAY_TASK_ID

# brightness_factor_idx=$((grid_index % num_brightness_factors))
# normtype_idx=$((grid_index / num_brightness_factors))

# brightness_factor=${brightness_factors[$brightness_factor_idx]}
# normtype=${normtypes[$normtype_idx]}
# detach_normtype=${normtype_detach[$((normtype_idx % 2))]}

# # Load the pre-generated random configurations
# export GRID_INDEX=$grid_index
# export BRIGHTNESS_FACTOR=$brightness_factor
# export NORMTYPE=$normtype
# export HOMEOSTASIS=0  # Fixed to 0
# export LAMBDA_HOMEOS=1 # Fixed to 1 But not functional because homeostasis is deactivated
# export NORMTYPE_DETACH=$detach_normtype
# export SHUNTING=0
# export HOMEOSTASIS_GRADIENT=0

# # Submit random jobs with the fixed set of random parameters
# sbatch --export=ALL run_random_jobs.sh
