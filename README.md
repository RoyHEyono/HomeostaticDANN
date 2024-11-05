# HomeostaticDANN

**HomeostaticDANN** (Homeostatic Dale's Artificial Neural Network) is an implementation of a specialized artificial neural network incorporating homeostasis-inspired mechanisms. This project is grounded in Dale's Principle, combined with domain adaptation techniques to improve network robustness and stability. The model aims to enhance the adaptability and resilience of neural networks under variable conditions, balancing network excitability and inhibition inspired by biological systems.

## Work in Progress

**This repository is a work in progress.** Please note that the code, features, and documentation may change as development continues. I welcome your thoughts and comments on the project.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/HomeostaticDANN.git
   cd HomeostaticDANN
   ```

2. **Create and load the virtual environment**:
   To set up a virtual environment and install dependencies, run the following in the project’s root directory:

   ```bash
   ./make_venv.sh
   ```

   Then, load the virtual environment with:

   ```bash
   source ./load_venv.sh
   ```

## Usage

HomeostaticDANN provides two implementations: a dense homeostatic network and an RNN-based homeostatic network. You can find both models in the `HomeostaticDANN/models` directory.

Here’s an example command to run the Dense Homeostatic DANN with customizable hyperparameters:

```bash
python path/to/HomeostaticDANN/models/dense_mnist_task/src/train.py \
  --data.brightness_factor=<brightness_factor> \
  --train.dataset=<dataset_name> \
  --opt.use_sep_inhib_lrs=<use_sep_inhib_lrs> \
  --opt.lr=<learning_rate> \
  --opt.inhib_lrs.wei=<inhib_lr_wei> \
  --opt.inhib_lrs.wix=<inhib_lr_wix> \
  --opt.inhib_momentum=<inhib_momentum> \
  --opt.momentum=<momentum> \
  --train.batch_size=<batch_size> \
  --opt.lambda_homeo=<lambda_homeostasis> \
  --model.normtype=<norm_type> \
  --model.task_opt_inhib=<task_opt_inhib> \
  --model.homeostasis=<homeostasis> \
  --model.excitation_training=<excitation_training> \
  --model.hidden_layer_width=<hidden_layer_width> \
  --model.homeo_opt_exc=<homeo_opt_exc> \
  --opt.use_sep_bias_gain_lrs=<use_sep_bias_gain_lrs> \
  --exp.wandb_project=<wandb_project_name> \
  --model.implicit_homeostatic_loss=<implicit_homeostatic_loss> \
  --exp.wandb_entity=<wandb_entity_name> \
  --exp.use_wandb=<use_wandb>
  --exp.name=<experiment_name> \
  --exp.save_model=<save_model>
```

Replace the placeholder values (`<...>`) with appropriate values based on your experiment setup.

**Note**: Ensure you configure Weights & Biases (`wandb`) with your own account and entity to log and view results effectively.


