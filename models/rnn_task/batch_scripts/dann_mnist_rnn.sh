#!/usr/bin/env bash
#SBATCH --array=0-10  # 10 random configurations
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=64GB
#SBATCH --time=2:30:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/dann_mnist_rnn.%A.%a.out
#SBATCH --error=sbatch_err/dann_mnist_rnn.%A.%a.err
#SBATCH --job-name=dann_mnist_rnn

# Load environment
. /etc/profile
module load anaconda/3
conda activate ffcv_eg

# Run your training script with the specific parameters
python /home/mila/r/roy.eyono/HomeostaticDANN/models/rnn_task/src/train.py \
  --data.brightness_factor=0 \
  --model.is_dann=1 \
  --train.dataset='mnist' \
  --train.seed=$SLURM_ARRAY_TASK_ID \
  --opt.use_sep_inhib_lrs=1 \
  --opt.lr=0.1 \
  --opt.inhib_momentum=0 \
  --opt.momentum=0 \
  --train.batch_size=32 \
  --opt.lambda_homeo=1 \
  --model.normtype=0 \
  --model.homeo_opt_exc=0 \
  --opt.use_sep_bias_gain_lrs=0 \
  --exp.wandb_project=MNIST_RNN_NoDANN \
  --exp.wandb_entity=project_danns \
  --exp.use_wandb=1