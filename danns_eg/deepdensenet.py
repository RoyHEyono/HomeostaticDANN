import torch.nn as nn
import torch
import numpy as np 
import danns_eg
from danns_eg.conv import EiConvLayer, ConvLayer
from danns_eg.dense import EiDenseLayerHomeostatic
from danns_eg.sequential import Sequential
from danns_eg.normalization import CustomGroupNorm
from danns_eg.normalization import LayerNormalize, MeanNormalize
import wandb


class DeepDenseDANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, configs, scaler, num_layers=2, homeostasis=True, nonlinearity=None, detachnorm=0, is_dann=1, shunting=0):
        super(DeepDenseDANN, self).__init__()
        ni = max(1,int(hidden_size*0.1))
        self.num_layers = num_layers
        self.is_dann = is_dann
        self.detachnorm = detachnorm
        self.saved_activations = dict()
        self.scaler = scaler
        self.homeostasis = homeostasis
        self.hidden_size = hidden_size
        self.local_loss_fn = danns_eg.dense.LocalLossMean(configs.model.hidden_layer_width, nonlinearity_loss=configs.model.implicit_homeostatic_loss)
        self.local_loss_val = 0
        self.nonlinearity = nonlinearity
        self.wandb_log = configs.exp.use_wandb

        if self.is_dann:

            setattr(self, 'fc1', EiDenseLayerHomeostatic(input_size, hidden_size, homeostasis=homeostasis, nonlinearity=None,  ni=ni, split_bias=False, lambda_homeo=configs.opt.lambda_homeo , lambda_var=configs.opt.lambda_homeo_var, affine=configs.opt.use_sep_bias_gain_lrs,
                                        train_exc_homeo=configs.model.homeo_opt_exc, use_bias=True, implicit_loss=configs.model.implicit_homeostatic_loss, shunting=shunting, scaler=scaler))

            self.ln_shell = self.fc1.ln_shell

            # Hidden layers
            for i in range(2, self.num_layers + 1):
                setattr(self, f'fc{i}', EiDenseLayerHomeostatic(hidden_size, hidden_size, homeostasis=homeostasis, nonlinearity=None, ni=ni, split_bias=False, lambda_homeo=configs.opt.lambda_homeo, lambda_var=configs.opt.lambda_homeo_var, affine=configs.opt.use_sep_bias_gain_lrs,
                                        train_exc_homeo=configs.model.homeo_opt_exc, use_bias=True, implicit_loss=configs.model.implicit_homeostatic_loss, shunting=shunting, scaler=scaler))
                                        
            
            self.relu = nn.ReLU()
            
            setattr(self, f'fc_output', EiDenseLayerHomeostatic(hidden_size, output_size, nonlinearity=None, ni=max(1,int(output_size*0.1)), split_bias=False, lambda_homeo=configs.opt.lambda_homeo, lambda_var=configs.opt.lambda_homeo_var, affine=False,
                                        use_bias=True, output=True, scaler=scaler))
            # setattr(self, f'fc_output', nn.Linear(hidden_size, output_size, bias=True))
        else:
            setattr(self, 'fc1', nn.Linear(input_size, hidden_size, bias=True))

            # Hidden layers
            for i in range(2, self.num_layers + 1):
                setattr(self, f'fc{i}', nn.Linear(hidden_size, hidden_size, bias=True))
                                        
            
            self.relu = nn.ReLU()
            setattr(self, f'fc_output', nn.Linear(hidden_size, output_size, bias=True))

        self.evaluation_mode = False

        # if detachnorm:
        #     self.ln = LayerNormalize(hidden_size) if nonlinearity else None
        # else:
        #     self.ln = nn.LayerNorm(hidden_size, elementwise_affine=False) if nonlinearity else None

        self.ln = MeanNormalize(detachnorm) if nonlinearity else None
        
        self.configs = configs
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

            if self.homeostasis and torch.is_grad_enabled():
                self.local_loss_val = layer.local_loss_value
        
        return forward_hook

    def register_hooks(self):
        if self.is_dann:
            for i in range(1, self.num_layers + 1):
                setattr(self, f'fc{i}_hook', getattr(self, f'fc{i}').register_forward_hook(self.list_forward_hook(layername=f'fc{i}')))

    def remove_hooks(self):
        if self.is_dann:
            for i in range(1, self.num_layers + 1):
                getattr(self, f'fc{i}_hook').remove()
                getattr(self, 'ln_shell').remove_hook()

    def get_local_val(self):
        return self.local_loss_val

    def set_homeostasis(self, homeostasis):
        for i in range(1, self.num_layers + 1):
            getattr(self, f'fc{i}').homeostasis = homeostasis
    
    def set_ln(self, activate):
        self.nonlinearity = activate
        self.ln = MeanNormalize(self.detachnorm) if self.nonlinearity else None
        # self.ln = nn.LayerNorm(self.hidden_size, elementwise_affine=False) if activate else None
    
    def set_wandb(self, activate):
        self.wandb_log = activate
    
    def forward(self, x):
        self.x = x.clone()
        for i in range(1, self.num_layers + 1):
            x = getattr(self, f'fc{i}')(x)

            if self.nonlinearity is not None:
                x = self.ln(x)

            x = self.relu(x)

        x = getattr(self, f'fc_output')(x)
        return x

def net(p:dict, scaler):

    input_dim = 784
    num_class = 10
    width=p.model.hidden_layer_width
    
    if p.model.is_dann:
        if p.model.homeostasis:
            model = DeepDenseDANN(input_dim, width, num_class, configs=p, scaler=scaler, num_layers=1, homeostasis=p.model.homeostasis, nonlinearity=None, detachnorm=p.model.normtype_detach, shunting=p.model.shunting)
        else:
            model = DeepDenseDANN(input_dim, width, num_class, configs=p, scaler=scaler, num_layers=1, homeostasis=p.model.homeostasis, nonlinearity=p.model.normtype, detachnorm=p.model.normtype_detach, shunting=p.model.shunting)
        return model
        
    
    else:
        model = DeepDenseDANN(input_dim, width, num_class, configs=p, scaler=scaler, num_layers=2, homeostasis=p.model.homeostasis, nonlinearity=p.model.normtype, is_dann=p.model.is_dann)

    return model

