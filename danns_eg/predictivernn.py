import torch.nn as nn
import torch
import numpy as np 

from danns_eg.conv import EiConvLayer, ConvLayer
from danns_eg.drnn import EiRNNCell
from danns_eg.dense import EiDenseLayerHomeostatic
from danns_eg.sequential import Sequential
from danns_eg.normalization import CustomGroupNorm
import wandb

class prnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, configs, homeostasis=True, nonlinearity=None):
        super(prnn, self).__init__()

        print(f"Homeostasis is {homeostasis}")

        self.configs = configs
        
        self.ei_cell = EiRNNCell(28, hidden_size, lambda_homeo=configs.opt.lambda_homeo , lambda_var=configs.opt.lambda_homeo_var, exponentiated=None, 
                        learn_hidden_init=False, homeostasis=homeostasis, ni_i2h=0.1, ni_h2h=0.1)
        
        self.ei_cell_output = []
        self.fc_output = EiDenseLayerHomeostatic(hidden_size, output_size, nonlinearity=nn.Softmax(dim=1), ni=max(1,int(output_size*0.1)), split_bias=False, use_bias=True)
        self.evaluation_mode = False
        
        self.nonlinearity = nn.LayerNorm(hidden_size, elementwise_affine=False) if nonlinearity else None
        self.register_eval = False
        

        

    def list_forward_hook(self, layername):
        def forward_hook(layer, input, output):
            # get mean and variance of the output on axis 1 and append to output list
            mu = torch.mean(output, axis=-1).mean().item()
            # Second moment instead of variance
            var = torch.mean(output**2, axis=-1).mean().item()
            if self.configs.exp.use_wandb: 
                if self.register_eval:
                    wandb.log({f"eval_{layername}_mu":mu, f"eval_{layername}_var":var})
                else:
                    wandb.log({f"train_{layername}_mu":mu, f"train_{layername}_var":var})
        return forward_hook

    def register_hooks(self):
        self.ei_cell_hook = self.ei_cell.register_forward_hook(self.list_forward_hook(self.ei_cell_output))


    def remove_hooks(self):
        self.ei_cell_hook.remove()

    # clear all the output lists
    def clear_output(self):
        self.ei_cell_output = []

    # get loss values from all fc layers
    def get_local_loss(self):
        total_local_loss = 0
        total_local_loss = total_local_loss + getattr(self, 'ei_cell').local_loss_value
        return total_local_loss

    def reset_hidden(self, batch_size):
        self.ei_cell.reset_hidden(requires_grad=True, batch_size=batch_size)
    
    def forward(self, x, fwd_pass_cnt):
        x_rnn = self.ei_cell(x, fwd_pass_cnt)
        if self.nonlinearity is not None:
            x = self.nonlinearity(x_rnn)
            x = self.fc_output(x)
            return x, x_rnn
        x = self.fc_output(x_rnn)
        return x, x_rnn

def net(p:dict):

    input_dim = 784
    num_class = 10
    hidden=p.model.hidden_layer_width

    if p.model.is_dann:
        return prnn(input_dim, hidden, num_class, configs=p, homeostasis=p.model.homeostasis, nonlinearity=p.model.normtype)
    else:
        raise NotImplementedError