#!/usr/bin/env bash
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

# Load the pre-generated random configurations
export LAMBDA_HOMEOS=300
export HOMEOSTASIS=1  # Fixed to 1
export NORMTYPE=0  # Fixed to 0

# Submit random jobs with the fixed set of random parameters
sbatch --export=ALL run_random_jobs.sh
