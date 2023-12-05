#!/usr/bin/env bash
#SBATCH --array=0-5%6
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=5:30:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/lambda_homeostasis.%A.%a.out
#SBATCH --error=sbatch_err/lambda_homeostasis.%A.%a.err
#SBATCH --job-name=lambda_homeostasis

. /etc/profile
module load anaconda/3
conda activate ffcv_eg

SLURM_JOB_ID=${SLURM_ARRAY_TASK_ID}

# Params
lmbda=(0 1e-2 1e-1)

python train.py --opt.lr=0.1 --opt.wd=0.001 --opt.momentum=0.9 --opt.inhib_lrs.wei=0.0001 --opt.inhib_lrs.wix=0.001 --train.batch_size=512 --model.normtype=ln --exp.wandb_entity=project_danns --train.seed=5 --train.use_testset=True --exp.save_model=True --model.name=resnet9 --opt.lambda_homeo=${lmbda[SLURM_JOB_ID]} --opt.lambda_var=$1
