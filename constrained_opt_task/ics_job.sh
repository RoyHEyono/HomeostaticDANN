#!/usr/bin/env bash
#SBATCH --array=0-19%20
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=0:30:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/ics_output.%A.%a.out
#SBATCH --error=sbatch_err/ics_error.%A.%a.err
#SBATCH --job-name=ics_computation




python ics_analysis.py --seed $SLURM_ARRAY_TASK_ID --cfg_file $1 --ics_layer $2