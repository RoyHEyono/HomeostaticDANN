#!/usr/bin/env bash
#SBATCH --array=0-10%11
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=0:30:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/ics_output.%A.%a.out
#SBATCH --error=sbatch_err/ics_error.%A.%a.err
#SBATCH --job-name=brightness_evaluation



source /home/mila/r/roy.eyono/.conda/envs/ffcv_eg/bin/activate base

brightness_eval_arr=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)

bright_eval=${brightness_eval_arr[$SLURM_ARRAY_TASK_ID]}

python /home/mila/r/roy.eyono/HomeostaticDANN/models/dense_mnist_task/src/train.py --data.brightness_factor=0.75 --data.brightness_factor_eval=$bright_eval --train.dataset='fashionmnist' --opt.lr=0.01 --opt.wd=1e-6 --opt.inhib_momentum=0.5 --opt.momentum=0.9 --opt.inhib_lrs.wei=1e-2 --opt.inhib_lrs.wix=1 --train.batch_size=32 --exp.wandb_project=Luminosity_Brightness_Robustness --exp.wandb_entity=project_danns --exp.use_wandb=True