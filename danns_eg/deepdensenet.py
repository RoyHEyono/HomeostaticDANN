import torch.nn as nn
import torch
import numpy as np 

from danns_eg.conv import EiConvLayer, ConvLayer
from danns_eg.dense import EiDenseLayerHomeostatic
from danns_eg.sequential import Sequential
from danns_eg.normalization import CustomGroupNorm
from danns_eg.normalization import LayerNormalize
import wandb


class DeepDenseDANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, configs, num_layers=2, homeostasis=True, nonlinearity=None, detachnorm=0, is_dann=1, shunting=0):
        super(DeepDenseDANN, self).__init__()
        ni = max(1,int(hidden_size*0.1))
        self.num_layers = num_layers
        self.is_dann = is_dann
        self.detachnorm = detachnorm
        self.saved_activations = dict()

        if self.is_dann:

            setattr(self, 'fc1', EiDenseLayerHomeostatic(input_size, hidden_size, homeostasis=homeostasis, nonlinearity=CustomLayerNormBackward() if homeostasis else None,  ni=ni, split_bias=False, lambda_homeo=configs.opt.lambda_homeo , lambda_var=configs.opt.lambda_homeo_var, affine=configs.opt.use_sep_bias_gain_lrs,
                                        train_exc_homeo=configs.model.homeo_opt_exc, use_bias=True, implicit_loss=configs.model.implicit_homeostatic_loss, shunting=shunting))

            # Hidden layers
            for i in range(2, self.num_layers + 1):
                setattr(self, f'fc{i}', EiDenseLayerHomeostatic(hidden_size, hidden_size, homeostasis=homeostasis, nonlinearity=CustomLayerNormBackward() if homeostasis else None, ni=ni, split_bias=False, lambda_homeo=configs.opt.lambda_homeo, lambda_var=configs.opt.lambda_homeo_var, affine=configs.opt.use_sep_bias_gain_lrs,
                                        train_exc_homeo=configs.model.homeo_opt_exc, use_bias=True, implicit_loss=configs.model.implicit_homeostatic_loss, shunting=shunting))
                                        
            
            self.relu = nn.ReLU()
            setattr(self, f'fc_output', EiDenseLayerHomeostatic(hidden_size, output_size, nonlinearity=None, ni=max(1,int(output_size*0.1)), split_bias=False, lambda_homeo=configs.opt.lambda_homeo, lambda_var=configs.opt.lambda_homeo_var, affine=False,
                                        use_bias=True))
        else:
            setattr(self, 'fc1', nn.Linear(input_size, hidden_size, bias=True))

            # Hidden layers
            for i in range(2, self.num_layers + 1):
                setattr(self, f'fc{i}', nn.Linear(hidden_size, hidden_size, bias=True))
                                        
            
            self.relu = nn.ReLU()
            setattr(self, f'fc_output', nn.Linear(hidden_size, output_size, bias=True))

        self.evaluation_mode = False
        if detachnorm:
            self.nonlinearity = LayerNormalize(hidden_size) if nonlinearity else None
        else:
            self.nonlinearity = nn.LayerNorm(hidden_size, elementwise_affine=False) if nonlinearity else None
        self.configs = configs
        self.register_eval = False
    
    def list_forward_hook(self, layername):
        def forward_hook(layer, input, output):
            layer.name = layername
            self.saved_activations[layername] = output.detach()
            layer.saved_activations = self.saved_activations # This is to compute variance in backward


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
    
    def list_backward_hook(self, layername):
        def gradient_normalization_with_variance_hook(module, grad_input, grad_output):
            """
            Normalize gradients and compute the variance of the layer's forward activations.
            """

            # # Normalize the gradients: NEEDS TO BE TESTED
            # if grad_input[0] is not None:
            #     srqt_var = torch.sqrt((module.saved_activations['fc1']).var(1) + module.epsilon) # temporary, we should change this to something dynamic
            #     srqt_var = srqt_var.view(-1, 1)  # Shape becomes [32, 1]
            #     normalized_grad_input = ((grad_input[0] - grad_input[0].mean(dim=1, keepdim=True)) / srqt_var,)
            #     return normalized_grad_input
        
        return gradient_normalization_with_variance_hook

    def register_hooks(self):
        if self.is_dann:
            for i in range(1, self.num_layers + 1):
                setattr(self, f'fc{i}_hook', getattr(self, f'fc{i}').register_forward_hook(self.list_forward_hook(layername=f'fc{i}')))
                setattr(self, f'fc{i}_hook_backward', getattr(self, f'fc{i}').register_full_backward_hook(self.list_backward_hook(layername=f'fc{i}')))
            setattr(self, f'fc_output_hook_backward', getattr(self, f'fc_output').register_full_backward_hook(self.list_backward_hook(layername=f'fc{i}')))
            self.fc_output.saved_activations = self.saved_activations
            self.fc_output.name = 'fc_output'

    def remove_hooks(self):
        if self.is_dann:
            for i in range(1, self.num_layers + 1):
                getattr(self, f'fc{i}_hook').remove()
                getattr(self, f'fc{i}_hook_backward').remove()
                getattr(self, f'fc{i}').nonlinearity.remove_hook()
            getattr(self, f'fc_output_hook_backward').remove()

    # get loss values from all fc layers
    def get_local_loss(self):
        total_local_loss = 0
        if self.is_dann:
            for i in range(1, self.num_layers + 1):
                total_local_loss = total_local_loss + getattr(self, f'fc{i}').local_loss_value
            return total_local_loss / self.num_layers

        return total_local_loss

    def set_homeostatic_temp(self, lmbda):
        for i in range(1, self.num_layers + 1):
            getattr(self, f'fc{i}').set_lambda(lmbda)
    
    def forward(self, x):
        for i in range(1, self.num_layers + 1):
            pre_activation = getattr(self, f'fc{i}')(x)
            if self.nonlinearity is not None:
                x = self.nonlinearity(pre_activation)
                x = self.relu(x)
            else:
                x = self.relu(pre_activation)

        x = getattr(self, f'fc_output')(x)
        return x, pre_activation

def net(p:dict):

    layer_outputs = []

    input_dim = 784
    num_class = 10
    width=p.model.hidden_layer_width

    modules = []
    
    if p.model.is_dann:
        if p.model.homeostasis:
            model = DeepDenseDANN(input_dim, width, num_class, configs=p, num_layers=1, homeostasis=p.model.homeostasis, nonlinearity=None, detachnorm=p.model.normtype_detach, shunting=p.model.shunting)
        else:
            model = DeepDenseDANN(input_dim, width, num_class, configs=p, num_layers=1, homeostasis=p.model.homeostasis, nonlinearity=p.model.normtype, detachnorm=p.model.normtype_detach, shunting=p.model.shunting)
        return model
        
    
    else:
        model = DeepDenseDANN(input_dim, width, num_class, configs=p, num_layers=2, homeostasis=p.model.homeostasis, nonlinearity=p.model.normtype, is_dann=p.model.is_dann)

    return model


class CustomLayerNormBackward(nn.Module):
    def __init__(self):
        super(CustomLayerNormBackward, self).__init__()

        # Call the register hook function during initialization
        self.register_hook()
    
    def forward(self, x):
        self.x =  x.detach()
        
        with torch.no_grad():
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize the input
        x_norm = (x - mean) / torch.sqrt(var + 1e-5)
        
        return x_norm

    def backward_hook(self, module, grad_input, grad_output):
        g_out = grad_output[0]  # Gradient passed from next layer
        N, D = g_out.shape
        
        # Simulate stored statistics (replace with actual stats if available)
        x_mean = self.x.mean(dim=1, keepdim=True)  # Mock input mean
        x_var = self.x.var(dim=1, keepdim=True, unbiased=False)  # Mock input variance

        x_normalized = (self.x - x_mean) / torch.sqrt(x_var + 1e-5)
        
        # Normalize gradients
        g_centered = g_out - g_out.mean(dim=1, keepdim=True)
        g_decorrelated = g_centered - ((g_out * x_normalized).sum(dim=1, keepdim=True) * (x_normalized / D))
        g_scaled = g_decorrelated / torch.sqrt(x_var + 1e-5)
        
        return (g_scaled,)

    def register_hook(self):
        self.hook = self.register_full_backward_hook(self.backward_hook)
    
    def remove_hook(self):
        self.hook.remove()