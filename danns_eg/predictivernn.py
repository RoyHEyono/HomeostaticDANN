import torch.nn as nn
import torch
import numpy as np 

from danns_eg.conv import EiConvLayer, ConvLayer
from danns_eg.rnn import EiRNNCell
from danns_eg.dense import EiDenseLayerHomeostatic
from danns_eg.sequential import Sequential
from danns_eg.normalization import CustomGroupNorm

class prnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, homeostasis=True, nonlinearity=None):
        super(prnn, self).__init__()

        print(f"Homeostasis is {homeostasis}")
        
        self.ei_cell = EiRNNCell(784,hidden_size, nonlinearity=nn.LayerNorm(hidden_size) if nonlinearity=='ln_true' else None, exponentiated=None, 
                        learn_hidden_init=False, homeostasis=homeostasis, ni_i2h=0.1, ni_h2h=0.1)
        
        self.ei_cell_output = []
        self.fc1 = EiDenseLayerHomeostatic(hidden_size, output_size, nonlinearity=nn.Softmax(dim=1), ni=max(1,int(output_size*0.1)), split_bias=False, use_bias=True)
        self.evaluation_mode = False
        
        self.nonlinearity = nn.LayerNorm(hidden_size) if nonlinearity=='ln_true' else None
        

        

    def list_forward_hook(self, output_list):
        def forward_hook(layer, input, output):
            inh_output = torch.matmul(input[0], layer.Wix.T)
            if self.training:
                # get mean and variance of the output on axis 1 and append to output list
                mu = torch.mean(output, axis=-1)
                # Second moment instead of variance
                var = torch.mean(output**2, axis=-1)
                mu_inh = torch.mean(inh_output, axis=-1)
                var_inh = torch.mean(inh_output**2, axis=-1)
                output_list.append([torch.mean(mu).item(), torch.std(mu).item(), torch.mean(var).item(), torch.std(var).item(), torch.mean(mu_inh).item(), torch.mean(var_inh).item()])
            elif self.evaluation_mode:
                # get mean and variance of the output on axis 1 and append to output list
                mu = torch.mean(output, axis=-1).cpu().detach().numpy()
                #var = torch.var(output, axis=-1).cpu().detach().numpy()
                # Second moment instead of variance
                var = (torch.mean(output**2, axis=-1)).cpu().detach().numpy()
                mu_inh = (torch.mean(inh_output, axis=-1)).cpu().detach().numpy()
                var_inh = (torch.mean(inh_output**2, axis=-1)).cpu().detach().numpy()
                # zip the mean and variance together
                zipped_list = list(zip(mu, var, mu_inh, var_inh))
                output_list.extend(zipped_list)

            
        return forward_hook

    def register_hooks(self):
        self.ei_cell_hook = self.ei_cell.register_forward_hook(self.list_forward_hook(self.ei_cell_output))


    def remove_hooks(self):
        self.ei_cell.remove()

    # clear all the output lists
    def clear_output(self):
        self.ei_cell_output = []

    # get loss values from all fc layers
    def get_local_loss(self):
        # add all the losses
        if self.ei_cell.local_loss_value is None:
            return None
        total_loss = self.ei_cell.local_loss_value
        return total_loss

    def reset_hidden(self, batch_size):
        self.ei_cell.reset_hidden(requires_grad=True, batch_size=batch_size)
    
    def forward(self, x):
        x = self.ei_cell(x)
        # if self.nonlinearity is not None:
        #     x = self.nonlinearity(x)
        x = self.fc1(x)
        return x

def net(p:dict):

    input_dim = 784
    num_class = 10
    hidden=500

    if p.model.is_dann:
        return prnn(input_dim, hidden, num_class, homeostasis=p.model.homeostasis, nonlinearity=p.model.normtype)
    else:
        raise NotImplementedError