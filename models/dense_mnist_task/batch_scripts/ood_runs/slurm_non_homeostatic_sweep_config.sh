#!/usr/bin/env bash
#SBATCH --array=0-1
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

normtypes=(0 1)  # Varying normtypes

# Calculate grid parameters based on SLURM_ARRAY_TASK_ID
normtype_idx=$SLURM_ARRAY_TASK_ID
normtype=${normtypes[$normtype_idx]}

# Load the pre-generated random configurations
export NORMTYPE=$normtype
export HOMEOSTASIS=0  # Fixed to 0
export LAMBDA_HOMEOS=1 # Fixed to 1 But not functional because homeostasis is deactivated

# Submit random jobs with the fixed set of random parameters
sbatch --export=ALL run_random_jobs.sh
