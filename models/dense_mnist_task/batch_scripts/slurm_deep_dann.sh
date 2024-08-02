#!/usr/bin/env bash
#SBATCH --array=0-53%50
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=0:50:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/homeostatic_entropy_dann.%A.%a.out
#SBATCH --error=sbatch_err/homeostatic_entropy_dann.%A.%a.err
#SBATCH --job-name=nonhomeostatic_entropy_deep_dann_sml

. /etc/profile
module load anaconda/3
conda activate ffcv_eg

# Params
lr_arr=(0.001 0.01 0.1)
bf_arr=(0 0.75)
lr_wei_arr=(1e-5 1e-4 1e-3)
lr_wix_arr=(1e-2 1e-1 1)

batch_size=32

# Calculate lengths and indices
len1=${#lr_arr[@]}   # 3
len2=${#bf_arr[@]}   # 2
len3=${#lr_wei_arr[@]}  # 3
len4=${#lr_wix_arr[@]}  # 3

# Calculate total number of combinations
len123=$((len1*len2*len3))   # 18
idx4=$((SLURM_ARRAY_TASK_ID/len123)) # 0, 1, 2
idx123=$((SLURM_ARRAY_TASK_ID%len123)) # 0, 1, ... 17

len12=$((len1*len2))  # 6
idx3=$((idx123/len12)) # 0, 1, 2
idx12=$((idx123%len12)) # 0, 1, ... 5

idx2=$((idx12/len1))  # 0, 1
idx1=$((idx12%len1))  # 0, 1, 2

# Extract parameters based on calculated indices
lr=${lr_arr[$idx1]}
bf=${bf_arr[$idx2]}
lr_wei=${lr_wei_arr[$idx3]}
lr_wix=${lr_wix_arr[$idx4]}

python /home/mila/r/roy.eyono/HomeostaticDANN/models/dense_mnist_task/src/train.py \
  --data.brightness_factor=$bf \
  --train.dataset='fashionmnist' \
  --opt.use_sep_inhib_lrs=1 \
  --opt.lr=$lr \
  --opt.inhib_lrs.wei=$lr_wei \
  --opt.inhib_lrs.wix=$lr_wix \
  --opt.inhib_momentum=0 \
  --opt.momentum=0 \
  --train.batch_size=$batch_size \
  --opt.lambda_homeo=$1 \
  --model.normtype=$2 \
  --model.task_opt_inhib=0 \
  --model.homeostasis=0 \
  --model.homeo_opt_exc=0 \
  --opt.use_sep_bias_gain_lrs=0 \
  --exp.wandb_project=Luminosity_DeepDANN_NoMomentum \
  --exp.wandb_entity=project_danns \
  --exp.use_wandb=1
