#export
import types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from danns_eg.normalization import LayerNormalizeCustom, LayerNormalizeCustomFA, MeanNormalize
        
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


class EiDenseLayerDecoupledHomeostatic(BaseModule):
    """
    Class modeling a subtractive feed-forward inhibition layer
    """
    def __init__(self, n_input, ne, ni=0.1, nonlinearity=None,use_bias=True, split_bias=False, lambda_homeo=1, lambda_homeo_var=1, scaler=None, gradient_norm=False,
                 ln_feedback='full', init_weights_kwargs={"numerator":2, "ex_distribution":"lognormal", "k":1}):
        """
        ne : number of exciatatory outputs
        ni : number (argument is an int) or proportion (float (0,1)) of inhibtitory units
        """
        super().__init__()

        assert n_input >= ne, "Class doesn't support matrices where n_input < ne, yet"

        self.n_input = n_input
        self.n_output = ne
        self.nonlinearity = nonlinearity
        self.split_bias = split_bias
        self.use_bias = use_bias
        self.ne = ne
        self.lambda_homeo = lambda_homeo
        self.lambda_homeo_var = lambda_homeo_var
        self.loss_fn = self.LocalLossMean()
        self.loss_var_fn = self.LocalLossVar()
        self.ln_norm = torch.nn.LayerNorm(ne, elementwise_affine=False)
        self.gradient_alignment_val = 0
        self.output_alignment_val = 0
        self.relu = nn.ReLU()
        if isinstance(ni, float): self.ni = int(ne*ni)
        elif isinstance(ni, int): self.ni = ni
        
        self.scaler = scaler
        # self.apply_ln_grad = LayerNormalizeCustom(no_forward=True, no_backward=(not gradient_norm))
        self.weights = torch.rand(self.ne)
        self.weights = self.weights / self.weights.sum()
        self.apply_ln_grad = LayerNormalizeCustomFA(self.weights, no_backward=(not gradient_norm), ln_feedback=ln_feedback)
        self.mean_norm = LayerNormalizeCustomFA(self.weights, no_backward=(not gradient_norm), ln_feedback='center')
        self.ln_feedback = ln_feedback

        # to-from notation - W_post_pre and the shape is n_output x n_input
        self.Wex = nn.Parameter(torch.empty(self.ne,self.n_input), requires_grad=True)
        self.Wix = nn.Parameter(torch.empty(self.ni,self.n_input), requires_grad=True)
        self.Wei = nn.Parameter(torch.empty(self.ne,self.ni), requires_grad=False)
        self.Bix = nn.Parameter(torch.empty(self.ne,self.n_input), requires_grad=True)
        self.Bei = nn.Parameter(torch.empty(self.ne,self.ne), requires_grad=False)

        self.local_mean_loss_value = 0
        self.local_var_loss_value = 0
        self.local_loss_value = 0
        
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
        
        self.init_weights(**init_weights_kwargs)

    class LocalLossMean(nn.Module):
        def __init__(self):
            super().__init__()
            self.criterion = nn.MSELoss()
            
        def forward(self, output_projection, excitatory_output, lambda_mean=1):
            
            mean = torch.mean(output_projection, dim=1, keepdim=True)
            mse_mean_ground_truth_loss = self.criterion(output_projection, excitatory_output - excitatory_output.mean(keepdim=True, axis=1))
            
            mean_term = mean ** 2 
            return lambda_mean * ((mean_term).mean()), (mse_mean_ground_truth_loss).item()

    class LocalLossVar(nn.Module):
        def __init__(self):
            super().__init__()
            self.criterion = nn.MSELoss()
            
        def forward(self, output_projection, excitatory_output, ln, lambda_var=1):
            
            mean = torch.mean(output_projection, dim=1, keepdim=True)
            var = output_projection.var(dim=-1, unbiased=False)
            ln_ground_truth_loss = self.criterion(output_projection, ln(excitatory_output))
            
            var_term = (var-1) ** 2
            mean_term = mean ** 2
            # return lambda_var * ln_ground_truth_loss, (ln_ground_truth_loss).item()
            return lambda_var * ((mean_term + var_term).mean()), (ln_ground_truth_loss).item()
            # return lambda_var * var_term.mean(), (ln_ground_truth_loss).item()

    def gradient_alignment(self, z, z_d):

        if self.ln_feedback == 'fa_center':
            loss_ln_sum = self.relu(self.mean_norm(z, z_d)).sum()
        else:
            loss_ln_sum = self.relu(self.ln_norm(z)).sum()
        
        loss_homeo_sum = self.relu(self.apply_ln_grad(z, z_d)).sum()

        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'Wex' in name:
                    grad_true = torch.autograd.grad(loss_ln_sum, param, retain_graph=True)[0]
                    grad_homeo = torch.autograd.grad(loss_homeo_sum, param, retain_graph=True)[0]
                    cos_sim = F.cosine_similarity(grad_true.view(-1).unsqueeze(0), grad_homeo.view(-1).unsqueeze(0))

        return cos_sim

    def output_alignment(self, z, z_d):

        ln_output = self.relu(self.ln_norm(z))
        homeo_output = self.relu(self.apply_ln_grad(z, z_d))
        cos_sim = F.cosine_similarity(ln_output.view(-1).unsqueeze(0), homeo_output.view(-1).unsqueeze(0))

        return cos_sim

    @property
    def W(self):
        return self.Wex - torch.matmul(self.Wei, self.Wix)
    
    @property
    def B(self):
        return self.Bex - torch.matmul(self.Bei, self.Bix)

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

        W = Wex_np - Wei_np@Wix_np

        _, S, V_T = np.linalg.svd(W)
        V = V_T[:self.ne].T

        Bix_np = np.diag(S) @ V.T
        Bei_np = np.ones((self.ne,self.ne))/self.ne

        self.Bix.data = torch.from_numpy(Bix_np).float()
        self.Bei.data = torch.from_numpy(Bei_np).float()

    def forward(self, x):

        # Compute excitatory input by projecting x onto Wex
        self.hex = torch.matmul(x, self.Wex.T)

        # If bias is used, add it to the excitation output
        # if self.use_bias:
        #     self.hex = self.hex + self.b.T
        
        # Compute inhibitory input, but detach x to prevent gradients from flowing back to x
        self.hi = torch.matmul(x.detach(), self.Wix.T)
        
        # Compute inhibitory output
        self.inhibitory_output = torch.matmul(self.hi, self.Wei.T)
        
        # Set excitation output as hex (raw excitatory response)
        self.excitation_output = self.hex
        
        # If bias is used, add it to the excitation output
        if self.use_bias: 
            self.excitation_output = self.excitation_output + self.b.T
        
        # Compute final response by subtracting inhibitory output (detached, so no gradient flow)
        self.z = self.excitation_output - self.inhibitory_output.detach()

        # Add divisive variance here...
        self.b_hi = torch.matmul(x.detach(), self.Bix.T)**2
        self.z_d_squared = torch.matmul(self.b_hi, self.Bei.T)
        self.z_d = torch.sqrt(self.z_d_squared+1e-5)

        # TODO: Compute divisive inhibition gradient here...
        if torch.is_grad_enabled():
            local_loss_var, self.local_loss_value  = self.loss_var_fn(((self.hex.detach() + self.b.T.detach())-self.inhibitory_output)/self.z_d, 
                                                                    (self.hex.detach() + self.b.T.detach()), self.ln_norm, self.lambda_homeo_var)
            
            # Scale and backpropagate the local loss, updating only the inhibitory weights
            self.scaler.scale(local_loss_var).backward()

        self.z = self.z/self.z_d.detach()

        # TODO: Apply gradient alignment here on some arbitrary loss.
        if torch.is_grad_enabled():
            self.gradient_alignment_val = self.gradient_alignment(self.z, self.z_d.detach()).item()
            self.output_alignment_val = self.output_alignment(self.z, self.z_d.detach()).item()
        
        # Apply layer normalization (or a similar transformation) to self.z
        self.z = self.apply_ln_grad(self.z, self.z_d.detach()) 

        # Apply a non-linearity if defined, otherwise, use the linear response
        if self.nonlinearity is not None:
            self.h = self.nonlinearity(self.z)
        else:
            self.h = self.z

        # Return the final processed output
        return self.h


