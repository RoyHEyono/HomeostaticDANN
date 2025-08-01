import torch.nn as nn
import torch
from danns_eg.dense import EiDenseLayerMeanHomeostatic, EiDenseLayer
import wandb


class HomeostaticDenseDANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, scaler, wandb=False, num_layers=2, detachnorm=0, homeo_lambda=1):
        super(HomeostaticDenseDANN, self).__init__()
        ni = max(1,int(hidden_size*0.1))
        self.num_layers = num_layers
        self.detachnorm = not detachnorm
        self.scaler = scaler
        self.hidden_size = hidden_size
        self.local_loss_val = 0
        self.wandb_log = wandb
        self.homeo_lambda = homeo_lambda
        self.forward_hook_step = 0


        setattr(self, 'fc0', EiDenseLayerMeanHomeostatic(input_size, hidden_size, ni=ni, nonlinearity=None, use_bias=True, split_bias=False, lambda_homeo=self.homeo_lambda, scaler=self.scaler, gradient_norm=self.detachnorm))

        # Hidden layers
        for i in range(0, self.num_layers):
            setattr(self, f'fc{i+1}', EiDenseLayerMeanHomeostatic(hidden_size, hidden_size, ni=ni, nonlinearity=None, use_bias=True, split_bias=False, lambda_homeo=self.homeo_lambda, scaler=self.scaler, gradient_norm=self.detachnorm))
                                    
        self.relu = nn.ReLU()
        
        setattr(self, f'fc_output', EiDenseLayer(hidden_size, output_size, ni=max(1,int(output_size*0.1)), nonlinearity=None, use_bias=True, split_bias=False))
            

        self.register_eval = False
    
    def list_forward_hook(self, layername):
        def forward_hook(layer, input, output):
            # get mean and variance of the output on axis 1 and append to output list
            mu = torch.mean(output, axis=-1).mean().item()
            # Second moment instead of variance
            var = output.var(dim=-1, keepdim=True, unbiased=False).mean().item()

            if self.wandb_log and self.forward_hook_step%1==0: 
                if self.register_eval:
                    wandb.log({f"eval_{layername}_mu":mu, f"eval_{layername}_var":var}, commit=False)
                else:
                    wandb.log({f"train_{layername}_mu":mu, f"train_{layername}_var":var}, commit=False)

            if torch.is_grad_enabled():
                self.local_loss_val = layer.local_loss_value
        
        return forward_hook

    def register_hooks(self):
        for i in range(self.num_layers+1):
            setattr(self, f'fc{i}_hook', getattr(self, f'fc{i}').register_forward_hook(self.list_forward_hook(layername=f'fc{i}')))

    def remove_hooks(self):
        for i in range(self.num_layers + 1):
            getattr(self, f'fc{i}_hook').remove()

    def get_local_val(self):
        return self.local_loss_val
    
    def forward(self, x):
        for i in range(self.num_layers+1):
            x = getattr(self, f'fc{i}')(x)
            x = self.relu(x)

        x = getattr(self, f'fc_output')(x)
        self.forward_hook_step += 1
        return x

def net(p:dict, scaler):

    input_dim = 784
    num_class = 10
    width=p.model.hidden_layer_width
        
    model = HomeostaticDenseDANN(input_dim, width, num_class, wandb=p.exp.use_wandb, scaler=scaler, num_layers=1, detachnorm=p.model.normtype_detach, homeo_lambda=p.opt.lambda_homeo)


    return model

