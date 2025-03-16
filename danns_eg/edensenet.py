import torch.nn as nn
import torch
import numpy as np 
import danns_eg
from danns_eg.conv import EiConvLayer, ConvLayer
from danns_eg.dense import EiDenseLayer, EDenseLayer
from danns_eg.sequential import Sequential
from danns_eg.normalization import CustomGroupNorm
from danns_eg.normalization import LayerNormalize, MeanNormalize
import wandb


class EDenseNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, wandb=0, num_layers=2, nonlinearity=0, detachnorm=0):
        super(EDenseNet, self).__init__()
        ni = max(1,int(hidden_size*0.1))
        self.num_layers = num_layers
        self.detachnorm = detachnorm
        self.hidden_size = hidden_size
        self.local_loss_val = 0
        self.nonlinearity = nonlinearity
        self.wandb_log = wandb
        

        setattr(self, 'fc0', EDenseLayer(input_size, hidden_size, ni=ni, nonlinearity=None, use_bias=True, split_bias=False))

        # Hidden layers
        for i in range(self.num_layers):
            setattr(self, f'fc{i+1}', EDenseLayer(hidden_size, hidden_size, ni=ni, nonlinearity=None, use_bias=True, split_bias=False))
                                    
        
        self.relu = nn.ReLU()
        
        setattr(self, f'fc_output', EiDenseLayer(hidden_size, output_size, ni=max(1,int(output_size*0.1)), nonlinearity=None, use_bias=True, split_bias=False))

        self.evaluation_mode = False

        self.ln = MeanNormalize(detachnorm) if nonlinearity else None
        self.register_eval = False
    
    def list_forward_hook(self, layername):
        def forward_hook(layer, input, output):

            total_out = output

            # get mean and variance of the output on axis 1 and append to output list
            mu = torch.mean(total_out, axis=-1).mean().item()
            # Second moment instead of variance
            var = total_out.var(dim=-1, keepdim=True, unbiased=False).mean().item()

            if self.wandb_log: 
                if self.register_eval:
                    wandb.log({f"eval_{layername}_mu":mu, f"eval_{layername}_var":var})
                else:
                    wandb.log({f"train_{layername}_mu":mu, f"train_{layername}_var":var})
        
        return forward_hook

    def register_hooks(self):
        for i in range(self.num_layers):
            setattr(self, f'fc{i}_hook', getattr(self, f'fc{i}').register_forward_hook(self.list_forward_hook(layername=f'fc{i}')))

    def remove_hooks(self):
        for i in range(self.num_layers):
            getattr(self, f'fc{i}_hook').remove()

    def get_local_val(self):
        return self.local_loss_val
    
    def forward(self, x):
        for i in range(self.num_layers+1):
            x = getattr(self, f'fc{i}')(x)

            if self.nonlinearity!=0:
                x = self.ln(x)

            x = self.relu(x)

        x = getattr(self, f'fc_output')(x)
        return x

def net(p:dict):

    input_dim = 784
    num_class = 10
    width=p.model.hidden_layer_width

    model = EDenseNet(input_dim, width, num_class, wandb=p.exp.use_wandb, num_layers=1, nonlinearity=p.model.normtype, detachnorm=p.model.normtype_detach)

    return model

