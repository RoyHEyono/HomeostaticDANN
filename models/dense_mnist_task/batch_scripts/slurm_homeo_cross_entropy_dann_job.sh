#!/usr/bin/env bash
#SBATCH --array=0-899%50
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=0:50:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/homeostatic_entropy_dann.%A.%a.out
#SBATCH --error=sbatch_err/homeostatic_entropy_dann.%A.%a.err
#SBATCH --job-name=homeostatic_entropy_dann

. /etc/profile
module load anaconda/3
conda activate ffcv_eg



# Params
lr_arr=(0.001 0.01 0.1 1 10)
bf_arr=(0 0.5 0.75 1)
homeo_lmbda_arr=(0 0.001 0.01 0.1 1)
lr_wei_arr=(1e-4 1e-2 0.1)
lr_wix_arr=(0.1 0.5 1)

batch_size=32

len1=${#lr_arr[@]}
len2=${#bf_arr[@]}
len3=${#homeo_lmbda_arr[@]}
len4=${#lr_wei_arr[@]}
len5=${#lr_wix_arr[@]}

len1234=$((len1*len2*len3*len4)) #300
idx5=$((SLURM_ARRAY_TASK_ID/len1234)) #0,1,2
idx1234=$((SLURM_ARRAY_TASK_ID%len1234)) #0,1,....299

len123=$((len1*len2*len3)) #100
idx4=$((idx1234/len123)) #0,1,2
idx123=$((idx1234%len123)) #0,1...99

len12=$((len1*len2)) #20
idx3=$((idx123/len12)) #0,1,2,3,4
idx12=$((idx123%len12)) #0,1...19

idx2=$((idx12/len1)) #0,1,2,3
idx1=$((idx1234%len1)) #0,1,2,3,4

lr=${lr_arr[$idx1]}
bf=${bf_arr[$idx2]}
lmbda=${homeo_lmbda_arr[$idx3]}
lr_wei=${lr_wei_arr[$idx4]}
lr_wix=${lr_wix_arr[$idx5]}

# lenL=${#lamda_arr[@]}
# lidx=$((SLURM_ARRAY_TASK_ID%lenL))
# pidx=$((SLURM_ARRAY_TASK_ID/lenL))
python /home/mila/r/roy.eyono/HomeostaticDANN/models/dense_mnist_task/src/train.py --data.brightness_factor=$bf --train.dataset='fashionmnist' --opt.use_sep_inhib_lrs=1 --opt.lr=$lr --opt.inhib_lrs.wei=$lr_wei --opt.inhib_lrs.wix=$lr_wix --opt.inhib_momentum=0.9 --opt.momentum=0.9 --model.homeostasis=1 --train.batch_size=$batch_size --opt.lambda_homeo=$lmbda --model.task_opt_inhib=1 --opt.use_sep_bias_gain_lrs=1 --exp.wandb_project=Luminosity --exp.wandb_entity=project_danns --exp.use_wandb=1
