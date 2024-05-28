#!/usr/bin/env bash
#SBATCH --array=0-899%50
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=0:30:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/vanilla_dann.%A.%a.out
#SBATCH --error=sbatch_err/vanilla_dann.%A.%a.err
#SBATCH --job-name=homeostatic_dann

. /etc/profile
module load anaconda/3
conda activate ffcv_eg



# Params
lr_arr=(0.001 0.01 0.1 1 10)
lr_wei_arr=(1e-4 1e-2 0.1 0.5)
lr_wix_arr=(1e-4 1e-2 0.1 0.5 1)
momentum_inhib_arr=(0 0.5 0.9)
momentum_arr=(0.5 0.75 0.9)
#wd_arr=(1e-6 1e-4 1e-3)

batch_size=32

len1=${#lr_arr[@]}
len2=${#lr_wei_arr[@]}
len3=${#lr_wix_arr[@]}
len4=${#momentum_inhib_arr[@]}
len5=${#momentum_arr[@]}

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
lr_wei=${lr_wei_arr[$idx2]}
lr_wix=${lr_wix_arr[$idx3]}
momentum_inhib=${momentum_inhib_arr[$idx4]}
momentum=${momentum_arr[$idx5]}

# lenL=${#lamda_arr[@]}
# lidx=$((SLURM_ARRAY_TASK_ID%lenL))
# pidx=$((SLURM_ARRAY_TASK_ID/lenL))
echo $idx1,$idx2,$idx3,$idx4,$idx5
echo $lr,$lr_wei,$lr_wix,$momentum_inhib,$momentum
python /home/mila/r/roy.eyono/HomeostaticDANN/models/dense_mnist_task/src/train.py --data.brightness_factor=$2 --train.dataset='fashionmnist' --opt.lr=$lr --opt.wd=$1 --opt.inhib_momentum=$momentum_inhib --opt.momentum=$momentum --opt.inhib_lrs.wei=$lr_wei --opt.inhib_lrs.wix=$lr_wix --train.batch_size=$batch_size --model.normtype=$3 --exp.wandb_project=Luminosity --exp.wandb_entity=project_danns --exp.use_wandb=True
