#!/usr/bin/env bash
#SBATCH --array=0-539%50
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=0:50:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/homeostatic_entropy_dann.%A.%a.out
#SBATCH --error=sbatch_err/homeostatic_entropy_dann.%A.%a.err
#SBATCH --job-name=homeostatic_entropy_deep_dann_sml

. /etc/profile
module load anaconda/3
conda activate ffcv_eg

# Params
lr_arr=(0.001 0.01 0.1)
bf_arr=(0 0.75)
homeo_lmbda_arr=(0.01 0.021544 0.1 0.21544 0.46416 1 1.5 4.6416 10 50)
lr_wei_arr=(1e-5 1e-4 1e-3)
lr_wix_arr=(1e-2 1e-1 1)

batch_size=32

len1=${#lr_arr[@]} # 3
len2=${#bf_arr[@]} # 2
len3=${#homeo_lmbda_arr[@]} # 3
len4=${#lr_wei_arr[@]} # 3
len5=${#lr_wix_arr[@]} # 3

len1234=$((len1*len2*len3*len4)) # 54
idx5=$((SLURM_ARRAY_TASK_ID/len1234)) # 0, 1, 2
idx1234=$((SLURM_ARRAY_TASK_ID%len1234)) # 0, 1, .... 53

len123=$((len1*len2*len3)) # 18
idx4=$((idx1234/len123)) # 0, 1, 2
idx123=$((idx1234%len123)) # 0, 1, ... 17

len12=$((len1*len2)) # 6
idx3=$((idx123/len12)) # 0, 1, 2
idx12=$((idx123%len12)) # 0, 1, ... 5

idx2=$((idx12/len1)) # 0, 1
idx1=$((idx12%len1)) # 0, 1, 2

lr=${lr_arr[$idx1]}
bf=${bf_arr[$idx2]}
lmbda=${homeo_lmbda_arr[$idx3]}
lr_wei=${lr_wei_arr[$idx4]}
lr_wix=${lr_wix_arr[$idx5]}

python /home/mila/r/roy.eyono/HomeostaticDANN/models/dense_mnist_task/src/train.py \
  --data.brightness_factor=$bf \
  --train.dataset='fashionmnist' \
  --opt.use_sep_inhib_lrs=1 \
  --opt.lr=$lr \
  --opt.inhib_lrs.wei=$lr_wei \
  --opt.inhib_lrs.wix=$lr_wix \
  --opt.inhib_momentum=0.9 \
  --opt.momentum=0.9 \
  --train.batch_size=$batch_size \
  --opt.lambda_homeo=$lmbda \
  --model.normtype=$1 \
  --model.task_opt_inhib=$2 \
  --model.homeostasis=1 \
  --model.homeo_opt_exc=0 \
  --opt.use_sep_bias_gain_lrs=0 \
  --exp.wandb_project=Luminosity_DeepDANN \
  --exp.wandb_entity=project_danns \
  --exp.use_wandb=1
