import torch.nn as nn
import torch
import numpy as np 

from danns_eg.conv import EiConvLayer, ConvLayer
from danns_eg.drnn import EiRNNCell, RNNCell
import danns_eg.drnn as drnn
from danns_eg.dense import EiDenseLayerHomeostatic
from danns_eg.sequential import Sequential
from danns_eg.normalization import CustomGroupNorm
from torch.cuda.amp import GradScaler
import torch.nn.functional as F
import wandb

class prnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, configs, scaler, homeostasis=True, nonlinearity=None, is_dann=1):
        super(prnn, self).__init__()

        print(f"Homeostasis is {homeostasis}")

        self.configs = configs
        self.num_layers = 3
        self.homeostasis = homeostasis
        self.is_dann = is_dann
        self.relu = nn.ReLU()
        self.hook_count = 0
        self.scaler = scaler
        self.local_loss_fn = drnn.LocalLossMean(configs.model.hidden_layer_width, nonlinearity_loss=configs.model.implicit_homeostatic_loss)

        if self.is_dann:
        
            self.ei_cell_0 = EiRNNCell(28, hidden_size, lambda_homeo=configs.opt.lambda_homeo , lambda_var=configs.opt.lambda_homeo_var, exponentiated=None, 
                            learn_hidden_init=False, homeostasis=homeostasis, ni_i2h=0.1, ni_h2h=0.1)
            self.ei_cell_1 = EiRNNCell(hidden_size, hidden_size, lambda_homeo=configs.opt.lambda_homeo , lambda_var=configs.opt.lambda_homeo_var, exponentiated=None, 
                            learn_hidden_init=False, homeostasis=homeostasis, ni_i2h=0.1, ni_h2h=0.1)
            self.ei_cell_2 = EiRNNCell(hidden_size, hidden_size, lambda_homeo=configs.opt.lambda_homeo , lambda_var=configs.opt.lambda_homeo_var, exponentiated=None, 
                            learn_hidden_init=False, homeostasis=homeostasis, ni_i2h=0.1, ni_h2h=0.1)
            
            
            self.fc_output = EiDenseLayerHomeostatic(hidden_size, output_size, nonlinearity=None, ni=max(1,int(output_size*0.1)), split_bias=False, use_bias=True)

        else:
            self.ei_cell_0 = RNNCell(28, hidden_size)
            self.ei_cell_1 = RNNCell(hidden_size, hidden_size)
            self.ei_cell_2 = RNNCell(hidden_size, hidden_size)
            self.fc_output = nn.Linear(hidden_size, output_size, bias=True)

        self.evaluation_mode = False
        self.ei_cell_output = []
        self.update_homeostatic_weights = False
        
        self.nonlinearity = nn.LayerNorm(hidden_size, elementwise_affine=False) if nonlinearity else None
        self.register_eval = False



    def list_forward_hook(self, layername):
        # Initialize stats_tracker, accumulated_loss, and hook_count for each layer
        if not hasattr(self, 'layer_data'):
            self.layer_data = {}

        if layername not in self.layer_data:
            self.layer_data[layername] = {
                'stats_tracker': {'mu': 0.0, 'var': 0.0, 'alpha': 0.99},
                'accumulated_loss': 0.0,
                'hook_count': 0,
            }

        def forward_hook(layer, input, output):
            # NOTE: This function is called 28 times for each layer in every mini-batch gradient update on sequential MNIST
            # NOTE: Ideally, you'd want to call the gradient update once after the 28 inputs 
            # Access layer-specific data
            layer_data = self.layer_data[layername]
            stats_tracker = layer_data['stats_tracker']
            alpha = stats_tracker['alpha']

            # Compute mean and second moment
            current_mu = torch.mean(output, axis=-1).mean().item()
            current_var = torch.mean(output**2, axis=-1).mean().item()

            # Update moving averages
            stats_tracker['mu'] = alpha * stats_tracker['mu'] + (1 - alpha) * current_mu
            stats_tracker['var'] = alpha * stats_tracker['var'] + (1 - alpha) * current_var

            # Increment hook count for this layer
            layer_data['hook_count'] += 1

            # Log to wandb periodically (e.g., every 1000 calls)
            log_frequency = 28
            if self.configs.exp.use_wandb and layer_data['hook_count'] % log_frequency == 0:
                if self.register_eval:
                    wandb.log({f"eval_{layername}_mu": stats_tracker['mu'], 
                            f"eval_{layername}_var": stats_tracker['var']})
                else:
                    wandb.log({f"train_{layername}_mu": stats_tracker['mu'], 
                            f"train_{layername}_var": stats_tracker['var']})

            # Compute gradients periodically (e.g., every 100 calls)
            update_frequency = 28 # This is based on sequentialMNIST
            if self.homeostasis and torch.is_grad_enabled():
                local_loss = self.local_loss_fn(output, 
                                                self.configs.opt.lambda_homeo, 
                                                self.configs.opt.lambda_homeo_var)
                layer_data['accumulated_loss'] += local_loss

                if layer_data['hook_count'] % update_frequency == 0:
                    for name, param in layer.named_parameters():
                        if param.requires_grad and ('Wix' in name or 'Wei' in name or 'Uix' in name or 'Uei' in name) and 'fc_output' not in name:
                            # Compute and assign gradient
                            param.grad = torch.autograd.grad(self.scaler.scale(layer_data['accumulated_loss']), param, retain_graph=True)[0]
                    
                    # Reset accumulated loss after gradient update
                    layer_data['accumulated_loss'] = 0.0
                    self.update_homeostatic_weights = True

        return forward_hook



    def set_optimizer(self, opt):
        self.opt = opt

    def register_hooks(self):
        if self.is_dann:
            for i in range(0, self.num_layers):
                setattr(self, f'ei_cell_{i}_hook', getattr(self, f'ei_cell_{i}').register_forward_hook(self.list_forward_hook(layername=f'ei_cell_{i}')))


    def remove_hooks(self):
        if self.is_dann:
            for i in range(0, self.num_layers):
                getattr(self, f'ei_cell_{i}_hook').remove()

    # get loss values from all fc layers
    def get_local_loss(self):
        total_local_loss = 0
        if self.is_dann:
            for i in range(0, self.num_layers):
                total_local_loss = total_local_loss + getattr(self, f'ei_cell_{i}').local_loss_value
            return total_local_loss / self.num_layers

        return total_local_loss

    def reset_hidden(self, batch_size):
        self.ei_cell_0.reset_hidden(requires_grad=True, batch_size=batch_size)
        self.ei_cell_1.reset_hidden(requires_grad=True, batch_size=batch_size)
        self.ei_cell_2.reset_hidden(requires_grad=True, batch_size=batch_size)
    
    def forward(self, x):
        x_rnn = self.ei_cell_0(x)
        for i in range(1, self.num_layers):
            pre_activation = getattr(self, f'ei_cell_{i}')(x_rnn)
            if self.nonlinearity is not None:
                x_rnn = self.nonlinearity(pre_activation)
                x_rnn = self.relu(x_rnn)
            else:
                x_rnn = self.relu(pre_activation)
        x = self.fc_output(x_rnn)

        if self.update_homeostatic_weights and self.homeostasis:
            self.scaler.step(self.opt)
            self.scaler.update()
            self.update_homeostatic_weights = False
        
        return x, x_rnn

def net(p:dict, scaler):

    input_dim = 784
    num_class = 10
    hidden=p.model.hidden_layer_width

    return prnn(input_dim, hidden, num_class, configs=p, scaler=scaler, homeostasis=p.model.homeostasis, nonlinearity=p.model.normtype, is_dann=p.model.is_dann)