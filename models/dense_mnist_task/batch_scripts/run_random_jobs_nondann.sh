#!/usr/bin/env bash
#SBATCH --array=0-49  # 100 random configurations
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=1:30:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/random_config_%A_%a.out
#SBATCH --error=sbatch_err/random_config_%A_%a.err
#SBATCH --job-name=nondann_sweep_config_run

# Load environment
. /etc/profile
module load anaconda/3
conda activate ffcv_eg

# Retrieve grid configuration parameters
grid_index=$GRID_INDEX
brightness_factor=$BRIGHTNESS_FACTOR
lambda_homeo=$LAMBDA_HOMEOS
homeostasis=$HOMEOSTASIS
normtype=$NORMTYPE

# Load random parameters from file
random_configs_file='random_nondann_configs.json'
random_index=$SLURM_ARRAY_TASK_ID
random_params=$(python -c "import json; import sys; f=open('$random_configs_file'); configs=json.load(f); f.close(); print(json.dumps(configs[$random_index]))")
lr=$(echo $random_params | python -c "import sys, json; config=json.load(sys.stdin); print(config['lr'])")
hidden_layer_width=$(echo $random_params | python -c "import sys, json; config=json.load(sys.stdin); print(config['hidden_layer_width'])")

# Run your training script with the specific parameters
python /home/mila/r/roy.eyono/HomeostaticDANN/models/dense_mnist_task/src/train.py \
  --data.brightness_factor=$brightness_factor \
  --model.is_dann=0 \
  --train.dataset='fashionmnist' \
  --opt.use_sep_inhib_lrs=1 \
  --opt.lr=$lr \
  --opt.inhib_momentum=0 \
  --opt.momentum=0 \
  --train.batch_size=32 \
  --opt.lambda_homeo=$lambda_homeo \
  --model.normtype=$normtype \
  --model.hidden_layer_width=$hidden_layer_width \
  --model.homeo_opt_exc=0 \
  --opt.use_sep_bias_gain_lrs=0 \
  --exp.wandb_project=Luminosity_NoDANN_ExhaustiveSearch \
  --exp.wandb_entity=project_danns \
  --exp.use_wandb=1
