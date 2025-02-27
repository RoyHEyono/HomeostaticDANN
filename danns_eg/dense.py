#export
import types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
        
class BaseModule(nn.Module):
    """
    Base class formalising the expected structure of modules 
    (e.g. the bias parameter is dependent on if using exponentiated
    gradient) and implementing the string representation. 
    """
    def __init__(self):
        super().__init__()
        self.n_input = None
        self.n_output = None
        self.nonlinearity = None
        self.spit_bias = None
    
    @property
    def input_shape(self): return self.n_input # reassess fter we have the rnns etc coded up

    @property
    def output_shape(self): return self.n_output

    @property
    def b(self):
        """
        Expected to be something like:
        ```
        if self.split_bias: return self.bias_pos + self.bias_neg
        else: return self.bias
        ```
        """
        raise NotImplementedError

    def init_weights(self, **args):
        """
        Expected to not include the bias, instead bias init in the __init__,
        """
        raise NotImplementedError
    
    def patch_init_weights_method(self, obj):
        """ obj should be a callable 

        For example:
        ```
        def normal_init(self, numerator=2):
            print("patched init method being used")
            nn.init.normal_(self.W, mean=0, std=np.sqrt((numerator / self.n_input)))
        
        l = DenseLayer(784,10, nonlinearity=None, exponentiated=False)
        l.patch_init_weights_method(normal_init)
        l.init_weights()
        ```
        """
        assert callable(obj)
        self.init_weights = types.MethodType(obj,self)

    def forward(self, *inputs):
        raise NotImplementedError

    @property
    def param_names(self): # Todo: list/ eg where this is used
        return [p[0] for p in self.named_parameters()]

    def __repr__(self):
        """
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/module.py#L1529
        # def extra_repr(self):
        """
        r  = ''
        r += str(self.__class__.__name__)+' '
        for key, param in self.named_parameters():
            r += key +' ' + str(list(param.shape))+' '
        if self.nonlinearity is None: r += 'Linear'
        else: r += str(self.nonlinearity.__name__)

        child_lines = []
        for key, module in self._modules.items():
            child_repr = "  "+repr(module)
            child_lines.append('(' + key + '): ' + child_repr)

        r += '\n  '.join(child_lines) #+ '\n'
        return r

class DenseLayer(BaseModule):
    def __init__(self, n_input, n_output, nonlinearity=None, use_bias=True, split_bias=False):
        """
        n_input:      input dimension
        n_output:     output dimension
        nonlinearity (callable or None): nonlinear activation function, if None then linear
        split_bias: bool, to split the bias into bias_pos + bias_neg
        use_bias (bool): If set to False, the layer will not learn an additive bias. Default: True
        """
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.nonlinearity = nonlinearity
        self.split_bias = split_bias 
        self.use_bias = use_bias

        self.W = nn.Parameter(torch.randn(n_output, n_input))
        if self.use_bias:
            if self.split_bias: # init and define bias as 0 depending on eg
                self.bias_pos = nn.Parameter(torch.ones(self.n_output,1))
                self.bias_neg = nn.Parameter(torch.ones(self.n_output,1)*-1)
            else:
                self.bias = nn.Parameter(torch.zeros(self.n_output, 1))
        else:
            self.register_parameter('bias', None)
            self.split_bias = False

        self.init_weights()

    @property
    def b(self):
        if self.split_bias: 
            return self.bias_pos + self.bias_neg
        else: 
            return self.bias
    
    def init_weights(self, numerator=2):
        """
        Initialises a Dense layer's weights (W) from a normal dist,
        and sets bias to 0.

        Note this is more a combination of Lecun init (just fan-in)
        and He init.

        References:
        https://arxiv.org/pdf/1502.01852.pdf
        http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf

        Use numerator=1 for sigmoid, numerator=2 for relu
        """
        nn.init.normal_(self.W, mean=0, std=np.sqrt((numerator / self.n_input)))

    def forward(self, x):
        """
        x is batch_dim x input_dim, 
        # todo - Transpose x as W is ne x input_dim 
        """
        # x is b x d, W is ne x d, b is ne x 1
        self.z = torch.mm(x, self.W.T) 
        if self.b: self.z = self.z + self.b.T 
        if self.nonlinearity is not None:
            self.h = self.nonlinearity(self.z)
        else:
            self.h = self.z
        return self.h

class EiDenseLayer(BaseModule):
    """
    Class modeling a subtractive feed-forward inhibition layer
    """
    def __init__(self, n_input, ne, ni=0.1, nonlinearity=None,use_bias=True, split_bias=False,
                 init_weights_kwargs={"numerator":2, "ex_distribution":"lognormal", "k":1}):
        """
        ne : number of exciatatory outputs
        ni : number (argument is an int) or proportion (float (0,1)) of inhibtitory units
        """
        super().__init__()
        self.n_input = n_input
        self.n_output = ne
        self.nonlinearity = nonlinearity
        self.split_bias = split_bias
        self.use_bias = use_bias
        self.ne = ne
        if isinstance(ni, float): self.ni = int(ne*ni)
        elif isinstance(ni, int): self.ni = ni

        # to-from notation - W_post_pre and the shape is n_output x n_input
        self.Wex = nn.Parameter(torch.empty(self.ne,self.n_input))
        self.Wix = nn.Parameter(torch.empty(self.ni,self.n_input))
        self.Wei = nn.Parameter(torch.empty(self.ne,self.ni))
        
        # init and define bias as 0, split into pos, neg if using eg
        if self.use_bias:
            if self.split_bias: 
                self.bias_pos = nn.Parameter(torch.ones(self.n_output,1)) 
                self.bias_neg = nn.Parameter(torch.ones(self.n_output,1)*-1)
            else:
                self.bias = nn.Parameter(torch.zeros(self.n_output, 1))
        else:
            self.register_parameter('bias', None)
            self.split_bias = False
        
        try:
            self.init_weights(**init_weights_kwargs)
        except:
            pass
            #print("Warning: Error initialising weights with default init!")

    @property
    def W(self):
        return self.Wex - torch.matmul(self.Wei, self.Wix)

    @property
    def b(self):
        if self.split_bias: 
            return self.bias_pos + self.bias_neg
        else: 
            return self.bias
    
    def init_weights(self, numerator=2, ex_distribution="lognormal", k=1):
        """
        Initialises inhibitory weights to perform the centering operation of Layer Norm:
            Wex ~ lognormal or exponential dist
            Rows of Wix are copies of the mean row of Wex
            Rows of Wei sum to 1, squashed after being drawn from same dist as Wex.  
            k : the mean of the lognormal is k*std (as in the exponential dist)
        """
        def calc_ln_mu_sigma(mean, var):
            """
            Helper function: given a desired mean and var of a lognormal dist 
            (the func arguments) calculates and returns the underlying mu and sigma
            for the normal distribution that underlies the desired log normal dist.
            """
            mu_ln = np.log(mean**2 / np.sqrt(mean**2 + var))
            sigma_ln = np.sqrt(np.log(1 + (var /mean**2)))
            return mu_ln, sigma_ln

        target_std_wex = np.sqrt(numerator*self.ne/(self.n_input*(self.ne-1)))
        # He initialistion standard deviation derived from var(\hat{z}) = d * ne-1/ne * var(wex)E[x^2] 
        # where Wix is set to mean row of Wex and rows of Wei sum to 1.

        if ex_distribution =="exponential":
            exp_scale = target_std_wex # The scale parameter, \beta = 1/\lambda = std
            Wex_np = np.random.exponential(scale=exp_scale, size=(self.ne, self.n_input))
            Wei_np = np.random.exponential(scale=exp_scale, size=(self.ne, self.ni))
        
        elif ex_distribution =="lognormal":
            # here is where we decide how to skew the distribution
            mu, sigma = calc_ln_mu_sigma(target_std_wex*k,target_std_wex**2)
            Wex_np = np.random.lognormal(mu, sigma, size=(self.ne, self.n_input))
            Wei_np = np.random.lognormal(mu, sigma, size=(self.ne, self.ni))
        
        Wei_np /= Wei_np.sum(axis=1, keepdims=True)
        Wix_np = np.ones(shape=(self.ni,1))*Wex_np.mean(axis=0,keepdims=True)
        self.Wex.data = torch.from_numpy(Wex_np).float()
        self.Wix.data = torch.from_numpy(Wix_np).float()
        self.Wei.data = torch.from_numpy(Wei_np).float()

    def forward(self, x):
        """
        x is batch_dim x input_dim, 
        therefore x.T as W is ne x input_dim ??? Why I got error?
        """
        self.z = torch.matmul(x, self.W.T)
        # if self.b: self.z = self.z + self.b.T
        if self.use_bias: self.z = self.z + self.b.T
        if self.nonlinearity is not None:
            self.h = self.nonlinearity(self.z)
        else:
            self.h = self.z
        return self.h

class CustomLayerNormBackward(nn.Module):
    def __init__(self):
        super(CustomLayerNormBackward, self).__init__()

        # Call the register hook function during initialization
        self.register_hook()
        self.Wex_var = 1
        self.backward_computation = 0
    
    def forward(self, x):
        # NOTE: TEMP
        # self.x =  x.detach()
        
        # with torch.no_grad():
        #     mean = x.mean(dim=-1, keepdim=True)
        #     var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # # Normalize the input
        # x_norm = (x - mean) / torch.sqrt(var + 1e-5)
        
        # return x_norm

        # # self.x = x

        return x
    
    def setVar(self, variance):
        self.Wex_var = variance

    def backward_hook(self, module, grad_input, grad_output):
        g_out = grad_output[0]  # Gradient passed from next layer
        N, D = g_out.shape
        
        # # Simulate stored statistics (replace with actual stats if available)
        # x_mean = self.x.mean(dim=1, keepdim=True)  # Mock input mean
        # x_var = self.x.var(dim=1, keepdim=True, unbiased=False)  # Mock input variance

        # x_normalized = (self.x - x_mean) / torch.sqrt(x_var + 1e-5)
        
        if self.backward_computation:
            # Normalize gradients
            g_centered = g_out - g_out.mean(dim=1, keepdim=True)
            g_decorrelated = g_centered # NOTE: TEMP - ((g_out * x_normalized).sum(dim=1, keepdim=True) * (x_normalized / D))
            g_scaled = g_decorrelated / torch.sqrt(self.Wex_var + 1e-5) # NOTE: Maybe in homeostasis, this variance should be the variance of the excitatory component alone
            # g_scaled = g_decorrelated / torch.sqrt(self.excitatory_var + 1e-5)
            
            return (g_scaled,)
    
    def register_hook(self):
        self.hook = self.register_full_backward_hook(self.backward_hook)
    
    def remove_hook(self):
        self.hook.remove()
class LocalLossMean(nn.Module):
        def __init__(self, hidden_size, nonlinearity_loss=False):
            super(LocalLossMean, self).__init__()
            self.nonlinearity = nn.LayerNorm(hidden_size, elementwise_affine=False)
            self.nonlinearity_loss = nonlinearity_loss
            self.criterion = nn.MSELoss()
            #self.kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)

        def cosine_similarity_loss(self, output_1, output_2):
            return 1 - F.cosine_similarity(output_1.flatten(1), output_2.flatten(1), dim=1).mean()

        def stronger_cosine_loss(self, output_1, output_2):
            cosine_sim = F.cosine_similarity(output_1.flatten(1), output_2.flatten(1), dim=1)
            angle = torch.acos(torch.clamp(cosine_sim, -1 + 1e-7, 1 - 1e-7))
            return angle.mean()
            
        def forward(self, output_projection, excitatory_output, inhibitory_output, lambda_mean=1, lambda_var=1):
            
            mean = torch.mean(output_projection, dim=1, keepdim=True)
            var = torch.var(output_projection, dim=1, keepdim=True, unbiased=False)
            inhibitory_loss = self.cosine_similarity_loss(output_projection, self.nonlinearity(excitatory_output).detach())
            mean_ground_truth_loss = self.cosine_similarity_loss(inhibitory_output.mean(keepdim=True,axis=1), excitatory_output.mean(keepdim=True, axis=1))
            mse_mean_ground_truth_loss = self.criterion(output_projection, excitatory_output - excitatory_output.mean(keepdim=True, axis=1))
            
            mean_term = mean ** 2 # torch.log(1 + mean ** 2) # mean ** 2  # Shape: (batch_size,) torch.log(1 + mean ** 2)
            var_term = (var - 1) ** 2 # torch.log(1 + (var - 1) ** 2) # (var - 1) ** 2  # Shape: (batch_size,)
            
            # return lambda_mean * ((mean_term + var_term).mean()), (inhibitory_loss).item()
            return lambda_mean * ((mean_term).mean()), (mse_mean_ground_truth_loss).item()


class EiDenseLayerHomeostatic(BaseModule): # Need to decouple this into two layers
    """
    Class modeling a subtractive feed-forward inhibition layer
    """
    def __init__(self, n_input, ne, ni=0.1, homeostasis=False, nonlinearity=None,use_bias=True, split_bias=False, lambda_homeo=1, lambda_var=1, affine=False, train_exc_homeo=False,
                 implicit_loss=False, shunting=False, output=False, scaler=None, init_weights_kwargs={"numerator":2, "ex_distribution":"lognormal", "k":1}):
        """
        ne : number of exciatatory outputs
        ni : number (argument is an int) or proportion (float (0,1)) of inhibtitory units
        """
        super().__init__()
        self.n_input = n_input
        self.n_output = ne
        self.nonlinearity = nonlinearity
        self.split_bias = split_bias
        self.use_bias = use_bias
        self.ne = ne
        self.homeostasis = homeostasis
        self.lambda_homeo = lambda_homeo
        self.lambda_var = lambda_var
        self.loss_fn = LocalLossMean(self.ne, nonlinearity_loss=implicit_loss)
        self.affine = affine
        self.train_exc_homeo = train_exc_homeo
        if isinstance(ni, float): self.ni = int(ne*ni)
        elif isinstance(ni, int): self.ni = ni
        self.output = output
        self.scaler = scaler
        self.ln_shell = CustomLayerNormBackward()
        self.ln_shell.backward_computation = implicit_loss

        self.n_input_with_bias = self.n_input+1

        # to-from notation - W_post_pre and the shape is n_output x n_input
        self.Wex = nn.Parameter(torch.empty(self.ne,self.n_input), requires_grad=True)
        self.Wix = nn.Parameter(torch.empty(self.ni,self.n_input), requires_grad=True if self.homeostasis or self.output else False)
        self.Wei = nn.Parameter(torch.empty(self.ne,self.ni), requires_grad=True if self.homeostasis or self.output else False)
        if shunting:
            self.alpha = nn.Parameter(torch.ones(size=(1, self.ni))) # row vector
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.local_loss_value = 0
        self.epsilon =  1e-6
        self.divisive_inh = shunting

        self.initialize = False

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(self.ne), requires_grad=True)
            self.beta = nn.Parameter(torch.zeros(self.ne), requires_grad=True)
        
        # init and define bias as 0, split into pos, neg if using eg
        if self.use_bias:
            if self.split_bias: 
                self.bias_pos = nn.Parameter(torch.ones(self.n_output,1)) 
                self.bias_neg = nn.Parameter(torch.ones(self.n_output,1)*-1)
            else:
                self.bias = nn.Parameter(torch.zeros(self.n_output, 1))
        else:
            self.register_parameter('bias', None)
            self.split_bias = False
        
        #try:
        self.init_weights(**init_weights_kwargs)
        # except:
        #     pass
            #print("Warning: Error initialising weights with default init!")

    @property
    def W(self):
        return self.Wex - torch.matmul(self.Wei, self.Wix)

    @property
    def b(self):
        if self.split_bias: 
            return self.bias_pos + self.bias_neg
        else: 
            return self.bias

    def set_lambda(self, lmbda):
        self.lambda_homeo = lmbda
    
    def init_weights(self, numerator=2, ex_distribution="lognormal", k=1):
        """
        Initialises inhibitory weights to perform the centering operation of Layer Norm:
            Wex ~ lognormal or exponential dist
            Rows of Wix are copies of the mean row of Wex
            Rows of Wei sum to 1, squashed after being drawn from same dist as Wex.  
            k : the mean of the lognormal is k*std (as in the exponential dist)
        """
        def calc_ln_mu_sigma(mean, var):
            """
            Helper function: given a desired mean and var of a lognormal dist 
            (the func arguments) calculates and returns the underlying mu and sigma
            for the normal distribution that underlies the desired log normal dist.
            """
            mu_ln = np.log(mean**2 / np.sqrt(mean**2 + var))
            sigma_ln = np.sqrt(np.log(1 + (var /mean**2)))
            return mu_ln, sigma_ln

        target_std_wex = np.sqrt(numerator*self.ne/(self.n_input*(self.ne-1)))
        # He initialistion standard deviation derived from var(\hat{z}) = d * ne-1/ne * var(wex)E[x^2] 
        # where Wix is set to mean row of Wex and rows of Wei sum to 1.

        if ex_distribution =="exponential":
            exp_scale = target_std_wex # The scale parameter, \beta = 1/\lambda = std
            Wex_np = np.random.exponential(scale=exp_scale, size=(self.ne, self.n_input))
            Wei_np = np.random.exponential(scale=exp_scale, size=(self.ne, self.ni))
        
        elif ex_distribution =="lognormal":
            # here is where we decide how to skew the distribution
            mu, sigma = calc_ln_mu_sigma(target_std_wex*k,target_std_wex**2)
            Wex_np = np.random.lognormal(mu, sigma, size=(self.ne, self.n_input))
            Wei_np = np.random.lognormal(mu, sigma, size=(self.ne, self.ni))
        
        Wei_np /= Wei_np.sum(axis=1, keepdims=True)
        Wix_np = np.ones(shape=(self.ni,1))*Wex_np.mean(axis=0,keepdims=True)
        self.Wex.data = torch.from_numpy(Wex_np).float()
        self.Wix.data = torch.from_numpy(Wix_np).float()
        self.Wei.data = torch.from_numpy(Wei_np).float()

    def forward(self, x):
        """
        x is batch_dim x input_dim, 
        therefore x.T as W is ne x input_dim ??? Why I got error?
        """

        self.hex = torch.matmul(x, self.Wex.T)
        
        self.hi = torch.matmul(x, self.Wix.T)
        self.hei = torch.matmul(self.relu(self.hi), self.Wei.T)
        
        if self.homeostasis:
            self.z = self.hex
            if self.use_bias: self.z = self.z + self.b.T
            self.ln_shell.setVar(self.z.var(dim=1, keepdim=True, unbiased=False).detach())
            self.z = self.z - self.hei.detach()
            if self.affine: self.z = self.z # + self.beta.detach()
            self.z = self.ln_shell(self.z) # At this point, you should be getting the correct grad_output
        else:
            self.z = self.hex - self.hei
            if self.use_bias: self.z = self.z + self.b.T

        
        if self.homeostasis and not self.output and torch.is_grad_enabled():
            excitatory_local = torch.matmul(x, self.Wex.T).detach()
            if self.use_bias: excitatory_local = excitatory_local + self.b.T.detach()
            # Compute a local loss for Wei and Wix
            hi_local = torch.matmul(x.detach(), self.Wix.T)
            hei_local = torch.matmul(self.relu(hi_local), self.Wei.T)  # Only affects Wei
            output_local = excitatory_local - hei_local

            if self.affine:
                # output_local = self.gamma * output_local + self.beta
                output_local = output_local  # + self.beta

            local_loss, self.local_loss_value  = self.loss_fn(output_local, excitatory_local, hei_local, self.lambda_homeo, self.lambda_var)  # Define your local loss

            # Compute gradients for Wei and Wix without affecting preceding layers
            self.scaler.scale(local_loss).backward()

        if self.nonlinearity is not None:
            self.h = self.nonlinearity(self.z)
        else:
            self.h = self.z

        return self.h

        
def init_eidense_ICLR(layer):
    """ 
    Initialises an EiDense layer's weights as in original paper 
    (note just the inhib_iid_init=False). Use to patch init method

    See https://openreview.net/pdf?id=eU776ZYxEpz
    """
    target_std = np.sqrt(2*np.pi/(layer.n_input*(2*np.pi-1)))
    exp_scale = target_std # The scale parameter, \beta = 1/\lambda = std
    Wex_np = np.random.exponential(scale=exp_scale, size=(layer.ne, layer.n_input))
    if layer.ni == 1: # for example the output layer
        Wix_np = Wex_np.mean(axis=0,keepdims=True) # not random as only one int
        Wei_np = np.ones(shape = (layer.ne, layer.ni))/layer.ni
    else:
        # We consider wee ~ wie, and the inhib outputs are Wei <- 1/ni
        Wix_np = np.random.exponential(scale=exp_scale, size=(layer.ni, layer.n_input))
        Wei_np = np.ones(shape = (layer.ne, layer.ni))/layer.ni

    layer.Wex.data = torch.from_numpy(Wex_np).float()
    layer.Wix.data = torch.from_numpy(Wix_np).float()
    layer.Wei.data = torch.from_numpy(Wei_np).float()
    # nn.init.zeros_(layer.b) # no longer setting bias in init weights

class EiDenseWithShunt(EiDenseLayer): 
    """
    PLACEHOLDER: TO BE TESTED AND FINISHED!
    """
    def __init__(self, n_input, ne, ni=0.1, nonlinearity=None, exponentiated=False):
        super().__init__(n_input, ne, ni, nonlinearity, exponentiated)


    def init_weights(self, layer, c=None):
        """
        Initialisation for network with forward equations of:

        Z = (1/c + gamma) * g*\hat(z) +b

        Where:
            c is a constant, that protects from division by a small value
            gamma_k = \sum_j wei_kj * alpha_j \sum_i Wix_ji x_i
            alpha = ln(e^\rho +1)

        Init strategy is to initialise:
            alpha = 1-c/ne E[Wex] E[X], therefore
            rho = ln(e^{(1-c)/ne E[Wex] E[X]} -1)

        Note** todo alpha is not a parameter anymore, so need to change the forward
        methods!!  

        Assumptions:
            X ~ rectified half normal with variance =1, therefore
                E[x] = 1/sqrt(2*pi)
            E[Wex] is the same as std(Wex) and both are equal to:
                sigma = np.sqrt(self.numerator*layer.ne/(layer.n_input*(layer.ne-1)))
        """
        super().init_weights(layer)
        raise NotImplementedError 
        if c is None: c_np = (5**0.5-1) /2 # golden ratio 0.618....
        else: c_np = c

        e_wex = np.sqrt(self.numerator*layer.ne/(layer.n_input*(layer.ne-1)))
        e_x   = 1/np.sqrt(2*np.pi)
        rho_np = np.log(np.exp(((1-layer.c)/layer.ne*e_wex*e_x)) -1) # torch softplus is alternative
        
        layer.c.data = torch.from_numpy(c_np).float()
        layer.rho.data = torch.from_numpy(rho_np).float()

    def forward(self, x):
        """
        PLACEHOLDER: TO BE TESTED AND FINISHED!
        x is batch_dim x input_dim, 
        therefore x.T as W is ne x input_dim
        # expose z dot etc for the homeostatic losses
        """
        raise NotImplementedError
        self.x = x.T
        self.ze = self.Wex@self.x  # ne x batch
        self.zi = self.Wix@self.x  # ni x btch

        # ne x batch = ne x batch - nexni ni x batch
        self.z_hat = self.ze - self.Wei@self.zi
        self.exp_alpha = torch.exp(self.alpha) # 1 x ni

        # ne x batch = (1xni * ^ne^xni ) @ nix^btch^ +  nex1
        self.gamma = ((self.exp_alpha*self.Wei)@self.zi) + self.epsilon

        # ne x batch = ne x batch * ne x batch
        self.z_dot = (1/ self.gamma) * self.z_hat

        # ne x batch = nex1*ne x batch + nex1
        self.z = self.g*self.z_dot + self.b
        # batch x ne
        self.z = self.z      # return batch to be first axis

        self.mu_z_layer  = self.z_hat.mean(axis=0, keepdim=True) # 1 x batch
        self.std_z_layer = self.z_hat.std(axis=0, keepdim=True) # 1 x batch

        if self.nonlinearity is not None:
            self.h = self.nonlinearity(self.z)
        else:
            self.h = self.z.clone()

        # retaining grad for ngd calculations
        # if self.zi.requires_grad:
        #     self.zi.retain_grad()
        #     self.z.retain_grad()
        #     self.gamma.retain_grad()
        return self.h.T

if __name__ == "__main__":
    # code sum tests here 
    # In order for imports to work correctly:
    # run in the shell 'export PYTHONPATH=$PYTHONPATH:~/dann-rnns'
    
    l = DenseLayer(784,10, nonlinearity=None, exponentiated=False)
    print(l)

    # test the set init weights method
    def normal_init(self, numerator=2):
        print("patched init method being used")
        nn.init.normal_(self.W, mean=0, std=np.sqrt((numerator / self.n_input)))
    l.patch_init_weights_method(normal_init)
    l.init_weights()

    l = DenseLayer(784,10,nonlinearity=None, exponentiated=True)
    print(l)


    # now test EiDense
    ei_layer = EiDense(784,784,ni=0.1, nonlinearity=F.relu, exponentiated=False)
    ei_layer.init_weights()

    # test the init_eidense_ICLR
    ei_layer.patch_init_weights_method(init_eidense_ICLR)
    ei_layer.init_weights()

    # Test building a lager network
    from sequential import Sequential
    def build_dense_net(layerclass, input_dim=784, hidden_dim=200,
                    output_dim=10, n_hidden_layers=5):
        """
        Todo: pass a layerclass kwargs dict
        """
        modules = [layerclass(input_dim, hidden_dim, F.relu)]
        
        for i in range(n_hidden_layers):
            modules.append(layerclass(hidden_dim, hidden_dim, F.relu))
    
        modules += [layerclass(hidden_dim, output_dim, None)]
        
        return Sequential(modules)

    model = build_dense_net(DenseLayer)
    print(model)
