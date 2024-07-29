import torch.nn as nn
import torch
import numpy as np 

from danns_eg.conv import EiConvLayer, ConvLayer
from danns_eg.dense import EiDenseLayerHomeostatic
from danns_eg.sequential import Sequential
from danns_eg.normalization import CustomGroupNorm
import wandb


class DeepDenseDANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, configs, num_layers=2, homeostasis=True, nonlinearity=None):
        super(DeepDenseDANN, self).__init__()
        ni = max(1,int(hidden_size*0.1))
        self.num_layers = num_layers

        setattr(self, 'fc1', EiDenseLayerHomeostatic(input_size, hidden_size, homeostasis=homeostasis, ni=ni, split_bias=False, lambda_homeo=configs.opt.lambda_homeo, affine=configs.opt.use_sep_bias_gain_lrs,
                                     train_exc_homeo=configs.model.homeo_opt_exc, use_bias=True))

        # Hidden layers
        for i in range(2, self.num_layers + 1):
            setattr(self, f'fc{i}', EiDenseLayerHomeostatic(hidden_size, hidden_size, homeostasis=homeostasis, ni=ni, split_bias=False, lambda_homeo=configs.opt.lambda_homeo, affine=configs.opt.use_sep_bias_gain_lrs,
                                     train_exc_homeo=configs.model.homeo_opt_exc, use_bias=True))
                                     
        
        self.relu = nn.ReLU()
        setattr(self, f'fc_output', EiDenseLayerHomeostatic(hidden_size, output_size, nonlinearity=nn.Softmax(dim=1), ni=max(1,int(output_size*0.1)), split_bias=False, lambda_homeo=configs.opt.lambda_homeo, affine=False,
                                     use_bias=True))
        self.evaluation_mode = False
        self.nonlinearity = nn.LayerNorm(hidden_size) if nonlinearity else None
        self.configs = configs
        self.register_eval = False
    
    def list_forward_hook(self, layername):
        def forward_hook(layer, input, output):
            inh_output = torch.matmul(input[0], layer.Wix.T)
            # get mean and variance of the output on axis 1 and append to output list
            mu = torch.mean(output, axis=-1).mean().item()
            # Second moment instead of variance
            var = torch.mean(output**2, axis=-1).mean().item()
            mu_inh = torch.mean(inh_output, axis=-1).mean().item()
            var_inh = torch.mean(inh_output**2, axis=-1).mean().item()
            if self.configs.exp.use_wandb: 
                if self.register_eval:
                    wandb.log({f"eval_{layername}_mu":mu, f"eval_{layername}_var":var, f"eval_{layername}_inh_mu":mu_inh, f"eval_{layername}_inh_var":var_inh})
                else:
                    wandb.log({f"train_{layername}_mu":mu, f"train_{layername}_var":var, f"train_{layername}_inh_mu":mu_inh, f"train_{layername}_inh_var":var_inh})
        return forward_hook

    def register_hooks(self):

        for i in range(1, self.num_layers + 1):
            setattr(self, f'fc{i}_hook', getattr(self, f'fc{i}').register_forward_hook(self.list_forward_hook(layername=f'fc{i}')))

    def remove_hooks(self):
        for i in range(1, self.num_layers + 1):
            getattr(self, f'fc{i}_hook').remove()

    # get loss values from all fc layers
    def get_local_loss(self):
        total_local_loss = 0
        for i in range(1, self.num_layers + 1):
            total_local_loss = total_local_loss + getattr(self, f'fc{i}').local_loss_value
        return total_local_loss / self.num_layers

    def set_homeostatic_temp(self, lmbda):
        for i in range(1, self.num_layers + 1):
            getattr(self, f'fc{i}').set_lambda(lmbda)
    
    def forward(self, x):

        for i in range(1, self.num_layers + 1):
            x = getattr(self, f'fc{i}').forward(x)
            if self.nonlinearity is not None:
                x = self.nonlinearity(x)
            x = self.relu(x)

        x = getattr(self, f'fc_output').forward(x)
        return x

def net(p:dict):

    layer_outputs = []

    input_dim = 784
    num_class = 10
    width=800

    modules = []
    
    if p.model.is_dann:
        if p.model.homeostasis:
            model = DeepDenseDANN(input_dim, width, num_class, configs=p, num_layers=2, homeostasis=p.model.homeostasis, nonlinearity=None)
        else:
            model = DeepDenseDANN(input_dim, width, num_class, configs=p, num_layers=2, homeostasis=p.model.homeostasis, nonlinearity=p.model.normtype)
        return model
        
    
    else:
        modules.append(nn.Linear(784, 128, bias=True))
        modules.append(nn.Linear(128, num_class, bias=True))
    
    model = Sequential(modules).to(memory_format=torch.channels_last).cuda()

    

    return model