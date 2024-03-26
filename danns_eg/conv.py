#%% 
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from danns_eg.normalization import CustomGroupNorm

# from dense import DenseLayer

class HeConv2d_WeightInitPolicy():
    """
    Remember BaseWeightInitPolicy is basically just nn.Module
    """
    @staticmethod
    def init_weights(conv2d):
        """
        Args:
            conv2d - an instance of nn.Conv2d

        Note this is more a combination of Lecun init (just fan-in)
        and He init (numerator is 2 dues to relu).

        References:
        https://arxiv.org/pdf/1502.01852.pdf
        http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
        """
        fan_in = np.prod(conv2d.weight.shape[1:]) # we scale weights for each filter's activation
        target_std = np.sqrt((2 / fan_in))

        if conv2d.bias is not None:
            nn.init.zeros_(conv2d.bias)

        nn.init.normal_(conv2d.weight, mean=0, std=target_std)
class ConvLayer(nn.Module):
    """
    Standard Conv2d

    This is a clunky implementation just wrapping Conv2d for similarity with ei conv layers.
    By defining this way network classes and policies approach can be shared.
    """
    def __init__(self, in_channels, out_channels, kernel_size, nonlinearity = None,
                 weight_init_policy=HeConv2d_WeightInitPolicy(), input_shape=None,
                 **kwargs):
        conv2d_kwargs = {'bias':False, 'stride':1, 'padding':1, 'dilation':1,
                         'groups':1,'padding_mode':'zeros'}
        conv2d_kwargs.update(kwargs)

        super().__init__()
        self.nonlinearity = nonlinearity
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, **conv2d_kwargs)
        self.input_shape = input_shape

        self.weight_init_policy = weight_init_policy
        self.network_index = None  # this will be set by the Network class
        self.network_key = None  # the layer's key for network's ModuleDict

        self.init_weights()
    def forward(self, x):
        self.z = self.conv2d(x)
        if self.nonlinearity is not None:
            self.h = self.nonlinearity(self.z)
        else:
            self.h = self.z
        return self.h

    def update(self, **kwargs):
        self.update_policy.update(self, **kwargs)

    def init_weights(self, **kwargs):
        "Not sure if it is best to code this as be passing self.conv tbh"
        self.weight_init_policy.init_weights(self.conv2d, **kwargs)

    @property
    def output_shape(self):
        if self.input_shape is None:
            return None
        else:
            data = torch.rand(self.input_shape).unsqueeze(0)
            return self.forward(data).shape[1:]

    def extra_repr(self):
        return ""

    def __repr__(self):
        """
        Here we are hijacking torch from printing details
        of the weight init policies
        """
        #print(str(self.conv2d.weight.shape))
        return self.conv2d.__repr__()

class EiConvInit_WexMean:
    def __init__(self, numerator=2, wex_distribution="lognormal"):
        self.numerator = numerator
        self.wex_distribution = wex_distribution
    
    def init_weights(self, layer):
        # this stems from calculating var(\hat{z}) = d * ne-1/ne * var(wex)E[x^2] when
        # we have set Wix to mean row of Wex and Wei as summing to 1.
        # here we are intepreting the actuvation filters at one point as being a layer
        # so instance norm almost? 

        n_input = int(np.prod(layer.Wex.shape[1:]))
        ne = int(layer.Wex.shape[0]) # i think this corresponds to what we are after
        ni = int(layer.Wix.shape[0])

        target_std_wex = np.sqrt(self.numerator*ne/(n_input*(ne-1)))
        exp_scale = target_std_wex # The scale parameter, \beta = 1/\lambda = std
        
        if self.wex_distribution =="exponential":
            Wex_np = np.random.exponential(scale=exp_scale, size=(layer.Wex.shape))
            Wei_np = np.random.exponential(scale=exp_scale, size=(layer.Wei.shape))
        
        elif self.wex_distribution =="lognormal":
            
            def calc_ln_mu_sigma(mean, var):
                "Given desired mean and var returns ln mu and sigma"
                mu_ln = np.log(mean**2 / np.sqrt(mean**2 + var))
                sigma_ln = np.sqrt(np.log(1 + (var /mean**2)))
                return mu_ln, sigma_ln

            mu, sigma = calc_ln_mu_sigma(target_std_wex,target_std_wex**2)
            Wex_np = np.random.lognormal(mu, sigma, size=(layer.Wex.shape))
            Wei_np = np.random.lognormal(mu, sigma, size=(layer.Wei.shape))
        
        Wei_np /= Wei_np.sum(axis=1, keepdims=True)
        Wix_np = np.ones(shape=layer.Wix.shape)*Wex_np.mean(axis=0,keepdims=True)
        # check this broadcasts correctly... 

        layer.Wex.data = torch.from_numpy(Wex_np).float() 
        layer.Wix.data = torch.from_numpy(Wix_np).float() 
        layer.Wei.data = torch.from_numpy(Wei_np).float()
        try: nn.init.zeros_(layer.b)
        except AttributeError: pass # not bias being used

class LocalLossMean(nn.Module):
        def __init__(self):
            super(LocalLossMean, self).__init__()

        def forward(self, inputs, targets=None, lambda_mu=1, lambda_var=1):

            mean = torch.mean(inputs, dim=(1,2,3), keepdim=True)
            #std = torch.std(inputs, dim=(1,2,3), unbiased=False, keepdim=True)
            mean_squared = torch.var(inputs, dim=(1,2,3), unbiased=False, keepdim=True)
            #mean_squared = torch.mean(torch.square(inputs), dim=(1,2,3), keepdim=True)

            # Define the target values (zero mean and unit standard deviation)
            target_mean = torch.zeros(mean.shape, dtype=inputs.dtype, device=inputs.device)
            #target_var = torch.ones(std.shape, dtype=inputs.dtype, device=inputs.device)
            target_mean_squared = torch.ones(mean_squared.shape, dtype=inputs.dtype, device=inputs.device)

            criterion = nn.MSELoss()

            # Calculate the loss based on the L2 distance from the target values
            loss = lambda_mu * torch.sqrt(criterion(mean, target_mean))  + lambda_var * torch.sqrt(criterion(mean_squared, target_mean_squared))
            #loss = (lambda_mu * (mean - target_mean) ** 2) + (lambda_var * (std - target_var) ** 2)
            
            return loss.mean()

class EiConvLayer(nn.Module):
    def __init__(self, in_channels, e_channels, i_channels, e_kernel_size, i_kernel_size,
                 nonlinearity = None,bias=False, 
                 weight_init_policy = EiConvInit_WexMean(), **kwargs):
        """
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        no support yet for groups   

        decided to not subclass tensor for postive only
        https://discuss.pytorch.org/t/how-to-add-attributes-to-a-tensor/126936 
        https://discuss.pytorch.org/t/subclassing-torch-tensor/23754/4              
        """   
        super().__init__()
        conv2d_kwargs = {'stride':1, 'padding':1, 'dilation':1,
                         'groups':1,'padding_mode':'zeros'}
        conv2d_kwargs.update(kwargs)
        self.padding = conv2d_kwargs['padding']
        self.stride = conv2d_kwargs['stride']
        self.dilation = conv2d_kwargs['dilation']
        self.groups = conv2d_kwargs['groups']
        if bias: self.bias = nn.Parameter(torch.zeros(e_channels, 1,1))
        else: self.bias=None   
        self.nonlinearity = nonlinearity
        self.weight_init_policy = weight_init_policy
        #self.loss_fn = nn.MSELoss()
        self.loss_fn = LocalLossMean()
        self.local_loss_multiplier = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.p = None
        self.norm_layer = None
        self.local_loss_value = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.multiplier_lr = 1
        #self.swish_fn = nn.SiLU()
        
        # Fisher corrections are only correct for same e and i filter params 
        # Therefore set i_params to e_params
        #i_param_dict = conv2d_kwargs    
        #self.e_conv = nn.Conv2d(in_channels, e_channels, e_kernel_size, **conv2d_kwargs)
        #self.i_conv = nn.Conv2d(in_channels, i_channels, i_kernel_size, **i_param_dict)
        self.Wex = nn.Parameter(torch.randn(e_channels, in_channels, e_kernel_size, e_kernel_size))
        self.Wix = nn.Parameter(torch.randn(i_channels, in_channels, i_kernel_size, i_kernel_size))
        self.Wei = nn.Parameter(torch.randn(e_channels, i_channels))

        if 'p' in kwargs:
            self.p = kwargs['p']
            if self.p.model.normtype == "bn": self.norm_layer = nn.BatchNorm2d(e_channels)
            elif self.p.model.normtype == "ln": self.norm_layer = nn.GroupNorm(1,e_channels, affine=False)
            elif self.p.model.normtype == "c_ln": self.norm_layer = CustomGroupNorm(1, e_channels, affine=False)
            elif self.p.model.normtype == "c_ln_sub": self.norm_layer = CustomGroupNorm(1, e_channels, subtractive=True, affine=False)
            elif self.p.model.normtype == "c_ln_div": self.norm_layer = CustomGroupNorm(1, e_channels, divisive=True, affine=False)
            elif self.p.model.normtype.lower() == "none": self.norm_layer = None
            self.multiplier_lr = self.p.opt.lr

        
        if self.norm_layer is not None:
            for param in self.norm_layer.parameters():
                param.requires_grad = False
        
        self.init_weights()

    def forward(self,x):
        """
        Note this assumes kernel sizes for e and i are the same!
        (and also padding etc, but not checking for that atm)
        """
        assert torch.all(self.Wix >= 0)
        e_shape = self.Wex.shape
        Wex = self.Wex.flatten(start_dim=1)
        Wix = self.Wix.flatten(start_dim=1)
        
        weight = torch.reshape(Wex - torch.matmul(self.Wei, Wix),e_shape)

        conv2d_unnormalized = F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        if self.p.model.homeostasis and self.training:
            # Compute the error difference between the normalized and unnormalized conv2d
            local_loss = self.loss_fn(conv2d_unnormalized, lambda_mu=self.p.opt.lambda_homeo, lambda_var=self.p.opt.lambda_var)

            # Set the local loss value to the computed loss
            self.local_loss_value = nn.Parameter(torch.tensor(local_loss.item()), requires_grad=False)

            # Compute gradients for specific parameters
            for name, param in self.named_parameters():
                if param.requires_grad:
                    if 'Wei' in name or 'Wix' in name:
                        param.grad = torch.autograd.grad(local_loss, param, retain_graph=True)[0]

        if not self.p.model.homeostasis and self.norm_layer is not None:
            # Normalize if not homeostasis but norm is specified
            return self.norm_layer(conv2d_unnormalized)
        
        return conv2d_unnormalized

    def forward_old(self, x):
        #print(x.shape)
        self.e_act_map = self.e_conv(x)
        self.i_act_map = self.i_conv(x)
        
        # produce subtractive map
        self.subtractive_map = (self.Wei @ self.i_act_map.permute(2,3,1,0)).permute(3,2,0,1)
        
        # produce a divisve map 
        # self.gamma = self.Wei @ (torch.exp(self.alpha) * self.i_act_map).permute(2,3,1,0)
        # self.gamma = self.gamma.permute(3,2,0,1) + self.epsilon
        
        self.zhat = self.e_act_map - self.subtractive_map
        #self.z_dot = (1/ self.gamma) * self.zhat
        
        #self.z = self.g*self.z_dot + self.b
        self.z = self.zhat + self.b
            
        if self.nonlinearity is not None:
            self.h = self.nonlinearity(self.z)
        else:
            self.h = self.z
        return self.h
    
    def update(self, **kwargs):
        self.update_policy.update(self,**kwargs)
        
    def init_weights(self, **kwargs):
        self.weight_init_policy.init_weights(self, **kwargs)

    def extra_repr(self):
        return "Nonlinearity: "+str(self.nonlinearity.__name__)
    
    def __repr__(self):
        """
        Here we are hijacking torch from printing details 
        of the weight init policies
        
        You should make two reprs , one to orint these detaisl
        """
        return f"EiConvLayer {self.Wex.shape}"
        return f'e{self.e_conv.__repr__()} \n     i{self.i_conv.__repr__()}'


# %%
