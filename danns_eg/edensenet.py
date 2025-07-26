import torch.nn as nn
import torch
import numpy as np 
import danns_eg
from danns_eg.conv import EiConvLayer, ConvLayer
from danns_eg.dense import EiDenseLayer, EDenseLayer
from danns_eg.sequential import Sequential
from danns_eg.normalization import CustomGroupNorm
from danns_eg.normalization import LayerNormalizeCustom, MeanNormalize, DivisiveNormalize
import wandb

NO_NORMALIZE = 0
MEAN_NORMALIZE = 1
VAR_NORMALIZE = 2
LN_NORMALIZE = 3

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
        self.forward_hook_step = 0
        

        setattr(self, 'fc0', EDenseLayer(input_size, hidden_size, ni=ni, nonlinearity=None, use_bias=True, split_bias=False))

        # Hidden layers
        for i in range(self.num_layers):
            setattr(self, f'fc{i+1}', EDenseLayer(hidden_size, hidden_size, ni=ni, nonlinearity=None, use_bias=True, split_bias=False))
                                    
        
        self.relu = nn.ReLU()
        
        setattr(self, f'fc_output', EiDenseLayer(hidden_size, output_size, ni=max(1,int(output_size*0.1)), nonlinearity=None, use_bias=True, split_bias=False))

        self.evaluation_mode = False
        
        if nonlinearity == NO_NORMALIZE:
            self.ln = None
        elif nonlinearity == MEAN_NORMALIZE:
            self.ln = MeanNormalize(detachnorm)
        elif nonlinearity == VAR_NORMALIZE:
            self.ln = DivisiveNormalize(detachnorm)
        elif nonlinearity == LN_NORMALIZE:
            self.ln = LayerNormalizeCustom(detachnorm)
        else:
            self.ln = None
        
        self.register_eval = False
    
    def list_forward_hook(self, layername):
        def forward_hook(layer, input, output):

            total_out = output

            # get mean and variance of the output on axis 1 and append to output list
            mu = torch.mean(total_out, axis=-1).mean().item()
            # Second moment instead of variance
            var = total_out.var(dim=-1, keepdim=True, unbiased=False).mean().item()

            if self.wandb_log and self.forward_hook_step%1==0: 
                if self.register_eval:
                    wandb.log({f"eval_{layername}_mu":mu, f"eval_{layername}_var":var}, commit=False)
                else:
                    wandb.log({f"train_{layername}_mu":mu, f"train_{layername}_var":var}, commit=False)
        
        return forward_hook

    def register_hooks(self):
        for i in range(self.num_layers + 1):
            setattr(self, f'fc{i}_hook', getattr(self, f'fc{i}').register_forward_hook(self.list_forward_hook(layername=f'fc{i}')))

    def remove_hooks(self):
        for i in range(self.num_layers + 1):
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
        self.forward_hook_step += 1 
        return x

def net(p:dict):

    input_dim = 784
    num_class = 10
    width=p.model.hidden_layer_width
    mean_normalize =  p.model.normtype
    divisive_normalize = p.model.divisive_norm
    layer_normalize = p.model.layer_norm
    if mean_normalize:
        norm_value = MEAN_NORMALIZE
    elif divisive_normalize:
        norm_value = VAR_NORMALIZE
    elif layer_normalize:
        norm_value = LN_NORMALIZE
    else:
        norm_value = NO_NORMALIZE

    model = EDenseNet(input_dim, width, num_class, wandb=p.exp.use_wandb, num_layers=1, nonlinearity=norm_value, detachnorm=p.model.normtype_detach)

    return model

