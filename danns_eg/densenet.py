import torch.nn as nn
import torch
import numpy as np 

from danns_eg.conv import EiConvLayer, ConvLayer
from danns_eg.dense import EiDenseLayerHomeostatic
from danns_eg.sequential import Sequential
from danns_eg.normalization import CustomGroupNorm


class DenseDANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, homeostasis=True):
        super(DenseDANN, self).__init__()
        ni = max(1,int(hidden_size*0.1))
        self.fc1 = EiDenseLayerHomeostatic(input_size, hidden_size, homeostasis=homeostasis, ni=ni, split_bias=False, # true is EG
                                     use_bias=True)
        self.relu = nn.ReLU()
        self.fc2 = EiDenseLayerHomeostatic(hidden_size, hidden_size, homeostasis=homeostasis, ni=ni, split_bias=False, # true is EG
                                     use_bias=True)
        self.fc2_output = []
        self.fc3 = EiDenseLayerHomeostatic(hidden_size, hidden_size, homeostasis=homeostasis, ni=ni, split_bias=False, # true is EG
                                     use_bias=True)
        self.fc3_output = []
        self.fc4 = EiDenseLayerHomeostatic(hidden_size, hidden_size, homeostasis=homeostasis, ni=ni, split_bias=False, # true is EG
                                     use_bias=True)
        self.fc4_output = []
        self.fc5 = EiDenseLayerHomeostatic(hidden_size, output_size, nonlinearity=nn.Softmax(), ni=max(1,int(output_size*0.1)), split_bias=False, # true is EG
                                     use_bias=True)
        self.evalution_mode = False

        

    def list_forward_hook(self, output_list):
        def forward_hook(layer, input, output):
            if self.training:
                # get mean and variance of the output on axis 1 and append to output list
                mu = torch.mean(output, axis=-1)
                var = torch.var(output, axis=-1)
                output_list.append([torch.mean(mu).item(), torch.std(mu).item(), torch.mean(var).item(), torch.std(var).item()])
            elif self.evalution_mode:
                # get mean and variance of the output on axis 1 and append to output list
                mu = torch.mean(output, axis=-1).cpu().detach().numpy()
                var = torch.var(output, axis=-1).cpu().detach().numpy()
                # zip the mean and variance together
                zipped_list = list(zip(mu, var))
                output_list.extend(zipped_list)

            
        return forward_hook

    def register_hooks(self):
        self.fc2_hook = self.fc2.register_forward_hook(self.list_forward_hook(self.fc2_output))
        self.fc3_hook = self.fc3.register_forward_hook(self.list_forward_hook(self.fc3_output))
        self.fc4_hook = self.fc4.register_forward_hook(self.list_forward_hook(self.fc4_output))


    def remove_hooks(self):
        self.fc2_hook.remove()
        self.fc3_hook.remove()
        self.fc4_hook.remove()

    # clear all the output lists
    def clear_output(self):
        self.fc2_output = []
        self.fc3_output = []
        self.fc4_output = []

    # get loss values from all fc layers
    def get_local_loss(self):
        # add all the losses
        if self.fc1.local_loss_value is None:
            return None
        total_loss = self.fc1.local_loss_value + self.fc2.local_loss_value \
                    + self.fc3.local_loss_value + self.fc4.local_loss_value
        total_loss = total_loss / 4
        return total_loss
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        return x

def net(p:dict):

    layer_outputs = []

    input_dim = 784
    num_class = 10
    width=500

    modules = []
    
    if p.model.is_dann:
        model = DenseDANN(input_dim, width, num_class)
        return model
        
    
    else:
        modules.append(nn.Linear(784, 128, bias=True))
        modules.append(nn.Linear(128, num_class, bias=True))
    
    model = Sequential(modules).to(memory_format=torch.channels_last).cuda()

    

    return model