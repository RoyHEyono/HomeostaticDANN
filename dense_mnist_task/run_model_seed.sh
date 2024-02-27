#!/usr/bin/env bash
#SBATCH --array=0-4%5
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=0:30:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/ics_output.%A.%a.out
#SBATCH --error=sbatch_err/ics_error.%A.%a.err
#SBATCH --job-name=homeo_dann_seed



#python train.py --opt.lr=0.1 --opt.wd=1e-6 --opt.inhib_momentum=0.9 --opt.momentum=0.75 --opt.inhib_lrs.wei=1e-2 --opt.inhib_lrs.wix=1 --train.batch_size=32 --exp.name=non_homeo_dann_"$SLURM_ARRAY_TASK_ID" --train.seed=$SLURM_ARRAY_TASK_ID


python train.py --opt.lr=0.1 --opt.wd=1e-6 --opt.inhib_momentum=0.9 --opt.momentum=0.5 --opt.inhib_lrs.wei=0.5 --opt.inhib_lrs.wix=1e-4 --train.batch_size=32 --exp.name=homeo_dann_"$SLURM_ARRAY_TASK_ID" --train.seed=$SLURM_ARRAY_TASK_ID