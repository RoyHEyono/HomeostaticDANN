#!/usr/bin/env bash
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=0:30:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/exp_track_danns_norm_hparam_sweep.%A.%a.out
#SBATCH --error=sbatch_err/exp_track_danns_norm_hparam_sweep.%A.%a.err
#SBATCH --job-name=exp_track_danns_norm_hparam_sweep

. /etc/profile
module unload python
module load anaconda/3
conda activate ffcv_eg

cd ..
python train.py --opt.lr=$2 --opt.wd=$3 --opt.momentum=$4 --opt.inhib_lrs.wei=$5 --opt.inhib_lrs.wix=$6 --train.batch_size=$7 --model.normtype=$8 --exp.wandb_project=Normalization_DANN_Test --exp.wandb_entity=project_danns --exp.use_wandb=False --train.seed=$9 --train.use_testset=True --exp.save_model=True
