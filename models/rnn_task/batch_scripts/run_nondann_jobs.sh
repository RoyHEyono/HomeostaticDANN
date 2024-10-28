#!/usr/bin/env bash
#SBATCH --array=0-99  # 100 random configurations
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=2:30:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/random_config_%A_%a.out
#SBATCH --error=sbatch_err/random_config_%A_%a.err
#SBATCH --job-name=nondann_sweep_config_run

# Load environment
. /etc/profile
module load anaconda/3
conda activate ffcv_eg


# Load random parameters from file
random_configs_file='random_nondann_configs.json'
random_index=$SLURM_ARRAY_TASK_ID
random_params=$(python -c "import json; import sys; f=open('$random_configs_file'); configs=json.load(f); f.close(); print(json.dumps(configs[$random_index]))")
lr=$(echo $random_params | python -c "import sys, json; config=json.load(sys.stdin); print(config['lr'])")

# Run your training script with the specific parameters
python /home/mila/r/roy.eyono/HomeostaticDANN/models/rnn_task/src/train.py \
  --data.brightness_factor=0 \
  --model.is_dann=0 \
  --train.dataset='mnist' \
  --opt.use_sep_inhib_lrs=1 \
  --opt.lr=$lr \
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