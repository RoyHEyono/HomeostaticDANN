# export
import types
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint

class BaseRNNCell(nn.Module):
    """
    Class formalising the expected structure of RNNCells
    """
    def __init__(self):
        super().__init__()
        self.n_input = None
        self.n_hidden = None
        self.nonlinearity = None
        self.exponentiated = None  

    @property
    def input_shape(self): return self.n_input 
    # reassess these properties after we have the rnns etc coded up, hidden? 

    @property
    def output_shape(self): return self.n_output

    @property
    def b(self):
        """
        Expected to be something like:
        ```
        if self.exponentiated: return self.bias_pos + self.bias_neg
        else: return self.bias
        ```
        """
        raise NotImplementedError

    def init_weights(self, **args):
        # could possibly split into i2h_init, h2h_init and call here
        raise NotImplementedError
    
    def patch_init_weights_method(self, obj):
        """ obj should be a callable 
        """
        assert callable(obj)
        self.init_weights = types.MethodType(obj,self)

    def reset_hidden(self, batch_size, **args):
        raise NotImplementedError

    def forward(self, *inputs):
        raise NotImplementedError


    def extra_repr(self):
        r = ''
        # r += str(self.__class__.__name__)+' '
        for key, param in self.named_parameters():
            r += key + ' ' + str(list(param.shape)) + ' '
        return r

class RNNCell(BaseRNNCell):
    """
    Class representing a standard RNN cell:
        U : input weights of shape n_hidden x n_input
        W : recurrent weights of shape n_hidden x n_hidden
        h_t = f(Ux + Wht-1 + b)
    """
    def __init__(self, n_input, n_hidden, nonlinearity=None,
                 exponentiated=False, learn_hidden_init=False):
        """
        n_input  : input dimensionality
        n_hidden : self hidden state dimensionality
        nonlinearity : 
        exponentiated
        learn_hidden_init (bool) : 
        """
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.nonlinearity = nonlinearity
        self.exponentiated = exponentiated

        # self.network_index = None # 0 being first cell in stack
        self.U = nn.Parameter(torch.randn(n_hidden, n_input))
        self.W = nn.Parameter(torch.randn(n_hidden, n_hidden))
        self.h0 = nn.Parameter(torch.zeros(self.n_hidden, 1), 
                               requires_grad=learn_hidden_init)

        if self.exponentiated: # init and define bias as 0 depending on eg
            self.bias_pos = nn.Parameter(torch.ones(self.n_hidden,1))
            self.bias_neg = nn.Parameter(torch.ones(self.n_hidden,1)*-1)
        else:
            self.bias = nn.Parameter(torch.zeros(self.n_hidden, 1))

        self.init_weights(numerator=1) # NOTE: Changed the numerator here to 1

    @property
    def b(self):
        if self.exponentiated: return self.bias_pos + self.bias_neg
        else: return self.bias
    
    def init_weights(self, numerator =1/3, h2h_identity=False):
        """
        If h2h_identity = True, uses an identity matrix init for self.W, see
        A Simple Way to Initialize Recurrent Networks of Rectified Linear Units
        https://arxiv.org/abs/1504.00941
            "For IRNNs, in addition to the recurrent weights being initialized at identity,
            the non-recurrent weights are initialized with a random matrix, whose entries
            are sampled from a Gaussian distribution with mean of zero and standard deviation of 0.001"
        """
        if h2h_identity: nn.init.eye_(self.W)
        else: nn.init.normal_(self.W, mean=0, std=np.sqrt((numerator/self.n_hidden)))
        nn.init.normal_(self.U, mean=0, std=np.sqrt((numerator/self.n_input)))
        # and bias is initialised as 0s in the __init__ function

    def reset_hidden(self,requires_grad,batch_size):
        self.h = self.h0.repeat(1, batch_size)  # Repeat tensor along bath dim.

    def forward(self, x):
        """
        x: input of shape input_dim x batch_dim
           U is h x input_dim
           W is h x h
        """
        # h x bs = (h x input *  input x bs) + (h x h * h x bs) + h
        self.z = torch.mm(self.U, x.T) + torch.mm(self.W, self.h) + self.b
        if self.nonlinearity is not None:
            self.h = self.nonlinearity(self.z)
        else:
            self.h = self.z
        return self.h.T

def torch_rnn_init(self):
    """
    Initialization replicating pytorch's initialisation approach for RNNs.
    Intended as an arg to the patch_init_weights method of BaseRNNCell.
    Currently not compatible with self.exponentiated=True.

    All parameters are initialsed as
        init.uniform_(self.bias, -bound, bound)
        bound = 1 / math.sqrt(fan_in)

    Where fan_in is taken to be n_hidden for rnn cells. Assumes an RNN cell
    with W, U & b parameter tensors. Note that RNNCells in pytorch have two
    bias vectors - one for h2h and one for i2h. Wherease this init only assumes one.

    ## Init justification:
    I think this is a "good-bug" the pytorch devs kept around due to good 
    empircal performance. Details:
    
    https://soumith.ch/files/20141213_gplus_nninit_discussion.htm
    https://github.com/pytorch/pytorch/issues/57109
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L44-L48

    # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
    # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
    # https://github.com/pytorch/pytorch/issues/57109
    """
    assert self.exponentiated != True
    bound = 1 / math.sqrt(self.n_hidden)
    nn.init.uniform_(self.W, -bound, bound)
    nn.init.uniform_(self.U, -bound, bound)
    nn.init.uniform_(self.b, -bound, bound)

def h2h_hybrid_uniform(self,p=0.85):
    """
    W = (1-p)*Id + p*W(Uniform Init)
    """
    bound = 1 / math.sqrt(self.n_hidden)
    W_p_np = np.random.uniform(-bound, bound, size=(self.n_hidden, self.n_hidden))
    W_id_np = np.eye(self.n_hidden)
    W = self.p * W_p_np + (1-self.p) * W_id_np
    self.W.data = torch.from_numpy(W).float().to('cuda' if torch.cuda.is_available else 'cpu')

class LocalLossMean(nn.Module):
        def __init__(self, hidden_size, nonlinearity_loss=False):
            super(LocalLossMean, self).__init__()
            self.nonlinearity = nn.LayerNorm(hidden_size, elementwise_affine=False)
            self.nonlinearity_loss = nonlinearity_loss
            self.criterion = nn.MSELoss()
            #self.kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
            
        def forward(self, inputs, lambda_mean=1, lambda_var=1):

            if not self.nonlinearity_loss:
                #kl_loss_val = self.kl_loss(torch.log_softmax(inputs, dim=-1), torch.log_softmax(self.nonlinearity(inputs), dim=-1))
                #cosine_loss = 1 - F.cosine_similarity(inputs, self.nonlinearity(inputs), dim=-1).mean()
                mse = lambda_mean * self.criterion(inputs, self.nonlinearity(inputs).detach())
                return mse # + kl_loss_val + cosine_loss
            
            mean = torch.mean(inputs, dim=1, keepdim=True)
            mean_squared = torch.mean(torch.square(inputs), dim=1, keepdim=True)

            # Define the target values (zero mean and unit standard deviation)
            target_mean = torch.zeros(mean.shape, dtype=inputs.dtype, device=inputs.device)
            target_mean_squared = torch.ones(mean_squared.shape, dtype=inputs.dtype, device=inputs.device)

            # print(f"mean: {torch.mean(mean)}, var: {torch.mean(mean_squared - mean**2)}")
            
            
            # Calculate the loss based on the L2 distance from the target values
            loss = lambda_mean * ( self.criterion(mean, target_mean)  + lambda_var * self.criterion(mean_squared, target_mean_squared))
            #loss = lambda_homeo * (torch.sqrt(criterion(mean_squared, target_mean_squared)))
            
            return loss.mean()

class EiRNNCell(BaseRNNCell):
    """
    Class modelling a DANN-RNN with subtractive feedforward
    inhibition between timesteps and layers

    h_t = f( (U^ex - U^eiU^ix)x + (W^ex - W^eiW^ix)h_t-1 + b)
    """
    def __init__(self, n_input, ne, ni_i2h=0.1, ni_h2h=0.1, nonlinearity=None,
                 exponentiated=False, learn_hidden_init=False, homeostasis=False, lambda_homeo=1, lambda_var=1, affine=False, train_exc_homeo=False,
                 implicit_loss=False):
        """
        ne : number of excitatory units in the "main" hidden layer, h
        ni_i2h : number of inhib units between x_t and h_t
        ni_h2h : number of inhib units between h_t-1 and h_t

        Todo, pass in hidden reset policy too.
        """
        super().__init__()
        self.n_input = n_input
        self.n_hidden = ne
        self.ne = ne # redundant naming going on here with n_hidden
        self.nonlinearity = nonlinearity # could directly set init gain from this
        self.exponentiated = exponentiated
        self.homeostasis = homeostasis
        self.lambda_homeo = lambda_homeo
        self.lambda_var = lambda_var
        self.affine = affine
        self.train_exc_homeo = train_exc_homeo
        self.local_loss_value = 0
        if isinstance(ni_i2h, float): self.ni_i2h = int(ne*ni_i2h)
        elif isinstance(ni_i2h, int): self.ni_i2h = ni_i2h
        if isinstance(ni_h2h, float): self.ni_h2h = int(ne*ni_h2h)
        elif isinstance(ni_h2h, int): self.ni_h2h = ni_h2h

        # to-from notation - U_post_pre with right mult, so n_post x n_pre
        self.Uex = nn.Parameter(torch.empty(ne,n_input))
        self.Uix = nn.Parameter(torch.empty(self.ni_i2h,n_input))
        self.Uei = nn.Parameter(torch.empty(ne,self.ni_i2h))

        self.Wex = nn.Parameter(torch.empty(ne,ne))
        self.Wix = nn.Parameter(torch.empty(self.ni_h2h,ne))
        self.Wei = nn.Parameter(torch.empty(ne,self.ni_h2h))

        self.h0 = nn.Parameter(torch.zeros(self.n_hidden, 1), 
                               requires_grad=learn_hidden_init)

        if self.exponentiated: # init and define bias as 0 depending on eg
            self.bias_pos = nn.Parameter(torch.ones(self.n_hidden,1))
            self.bias_neg = nn.Parameter(torch.ones(self.n_hidden,1)*-1)
        else:
            self.bias = nn.Parameter(torch.zeros(self.n_hidden, 1))

        self.init_weights()

    def init_weights(self, numerator =1/3, ex_distribution="lognormal"):
        """
        numerator : 2 for he, 1/3 for pytorch, 1 for xavier

        Initialises inhibitory weights to perform the centering operation of Layer Norm:
            Wex ~ lognormal or exponential dist
            Rows of Wix are copies of the mean row of Wex
            Rows of Wei sum to 1, squashed after being drawn from same dist as Wex.  

        Target std var calc is different for wex (n hidden) and uex (n input), 
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

        # first init the recurrent weights 
        target_std_wex = np.sqrt(numerator*self.ne/(self.n_hidden*(self.ne-1)))
        # He initialistion standard deviation derived from var(\hat{z}) = d * ne-1/ne * var(wex)E[x^2] 
        # where Wix is set to mean row of Wex and rows of Wei sum to 1.
        if ex_distribution =="exponential":
            exp_scale = target_std_wex # The scale parameter, \beta = 1/\lambda = std
            Wex_np = np.random.exponential(scale=exp_scale, size=(self.ne, self.ne))
            Wei_np = np.random.exponential(scale=exp_scale, size=(self.ne, self.ni_h2h))
        
        elif ex_distribution =="lognormal":
            # here is where we decide how to skew the distribution
            mu, sigma = calc_ln_mu_sigma(target_std_wex,target_std_wex**2)
            Wex_np = np.random.lognormal(mu, sigma, size=(self.ne, self.ne))
            Wei_np = np.random.lognormal(mu, sigma, size=(self.ne, self.ni_h2h))
        Wei_np /= Wei_np.sum(axis=1, keepdims=True)
        Wix_np = np.ones(shape=(self.ni_h2h,1))*Wex_np.mean(axis=0,keepdims=True)

        # now repeat for input weights 
        target_std_uex = np.sqrt(numerator*self.ne/(self.n_input*(self.ne-1)))
        if ex_distribution =="exponential":
            exp_scale = target_std_uex # The scale parameter, \beta = 1/\lambda = std
            Uex_np = np.random.exponential(scale=exp_scale, size=(self.ne, self.n_input))
            Uei_np = np.random.exponential(scale=exp_scale, size=(self.ne, self.ni_i2h)) 
        
        elif ex_distribution =="lognormal": # here is where we decide how to skew the distribution
            mu, sigma = calc_ln_mu_sigma(target_std_uex,target_std_uex**2)
            Uex_np = np.random.lognormal(mu, sigma, size=(self.ne, self.n_input))
            Uei_np = np.random.lognormal(mu, sigma, size=(self.ne, self.ni_i2h))
        
        Uei_np /= Uei_np.sum(axis=1, keepdims=True)
        Uix_np = np.ones(shape=(self.ni_i2h,1))*Uex_np.mean(axis=0,keepdims=True)

        self.Wex.data = torch.from_numpy(Wex_np).float()
        self.Wix.data = torch.from_numpy(Wix_np).float()
        self.Wei.data = torch.from_numpy(Wei_np).float()
        self.Uex.data = torch.from_numpy(Uex_np).float()
        self.Uix.data = torch.from_numpy(Uix_np).float()
        self.Uei.data = torch.from_numpy(Uei_np).float()

    @property
    def W(self):
        return self.Wex - torch.matmul(self.Wei, self.Wix)

    @property
    def U(self):
        return self.Uex - torch.matmul(self.Uei, self.Uix)

    @property
    def b(self):
        if self.exponentiated: return self.bias_pos + self.bias_neg
        else: return self.bias

    def reset_hidden(self,requires_grad,batch_size):
        self.h = self.h0.repeat(1, batch_size)  # Repeat tensor along bath dim.

    def forward(self, x):
        """
        x: input of shape input_dim x batch_dim
           U is h x input_dim
           W is h x h
        """
        # h x bs = (h x input *  input x bs) + (h x h * h x bs) + h
        # print(self.U.shape, x.shape, self.W.shape, self.h.shape)
        self.z = torch.mm(self.U, x.T) + torch.mm(self.W, self.h) + self.b

        if self.nonlinearity is not None: self.h = self.nonlinearity(self.z)
        else: self.h = self.z
        

        return self.h.T

# export
class EiRNNCellWithShunt(EiRNNCell):
    """
    Class modelling a DANN-RNN with subtractive and divisive
    feedforward inhibition between timesteps and layers

    Todo: update equation here
    h_t = f( (U^ex - U^eiU^ix)x + (W^ex - W^eiW^ix)h_t-1 + b)

    Todo: read up on the details of BN/LN across layers vs time,
    I remeber the first paper only managed to apply over depth (i think)
    for some reason.
    """
    def __init__(self, n_input, ne, ni_i2h=0.1, ni_h2h=0.1, nonlinearity=None,
                 exponentiated=False, learn_hidden_init=False):
        """
        ne : number of excitatory units in the "main" hidden layer, h
        ni_i2h : number of inhib units between x_t and h_t
        ni_h2h : number of inhib units between h_t-1 and h_t
        """

        super().__init__(n_input, ne, ni_i2h, ni_h2h,
                        nonlinearity,learn_hidden_init)

        self.U_alpha = nn.Parameter(torch.ones(size=(1, ni_i2h))) # row vector
        self.W_alpha = nn.Parameter(torch.ones(size=(1, ni_h2h)))
        self.U_g     = nn.Parameter(torch.ones(size=(ne,1)))
        self.W_g     = nn.Parameter(torch.ones(size=(ne,1)))

        self.epsilon = 1e-8

    def forward(self, x):
        """
        x: input of shape input_dim x batch_dim
           U is h x input_dim
           W is h x h
        """
        self.x = x.T

        self.W_ze = self.Wex @ self.x  # ne x batch
        self.W_zi = self.Wix @ self.x  # ni x btch
        # ne x batch = ne x batch - nexni ni x batch
        self.W_z_hat = self.W_ze - self.Wei @ self.W_zi
        self.W_exp_alpha = torch.exp(self.W_alpha)  # 1 x ni
        # ne x batch = (1xni * ^ne^xni ) @ nix^btch^ +  nex1
        self.W_gamma = ((self.W_exp_alpha * self.Wei) @ self.W_zi) + self.epsilon
        # ne x batch = ne x batch * ne x batch
        self.W_z = (1 / self.W_gamma) * self.W_z_hat
        # ne x batch = nex1*ne x batch + nex1
        self.W_z = self.W_g * self.W_z


        self.U_ze = self.Uex @ self.x  # ne x batch
        self.U_zi = self.Uix @ self.x  # ni x btch
        # ne x batch = ne x batch - nexni ni x batch
        self.U_z_hat = self.U_ze - self.Uei @ self.U_zi
        self.U_exp_alpha = torch.exp(self.U_alpha)  # 1 x ni
        # ne x batch = (1xni * ^ne^xni ) @ nix^btch^ +  nex1
        self.U_gamma = ((self.U_exp_alpha * self.Uei) @ self.U_zi) + self.epsilon
        # ne x batch = ne x batch * ne x batch
        self.U_z = (1 / self.U_gamma) * self.U_z_hat
        # ne x batch = nex1*ne x batch + nex1
        self.U_z = self.U_g * self.U_z

        self.z = self.U_z + self.W_z + self.b
        if self.nonlinearity is not None:
            self.h = self.nonlinearity(self.z)
        else:
            self.h = self.z.clone()

        # retaining grad for ngd calculations
        if self.W_zi.requires_grad or self.U_zi.requires_grad:
            self.W_zi.retain_grad()
            self.U_zi.retain_grad()
            self.z.retain_grad()
            self.W_gamma.retain_grad()
            self.U_gamma.retain_grad()

        return self.h.T

# export
if __name__ == "__main__":
    # In order for imports to work correctly:
    # run in the shell 'export PYTHONPATH=$PYTHONPATH:~/dann-rnns'
    
    cell = RNNCell(784,200, nonlinearity=None, exponentiated=None, learn_hidden_init=False)
    print(cell)

    ei_cell = EiRNNCell(784,200, nonlinearity=None, exponentiated=None, 
                        learn_hidden_init=False, ni_i2h=0.1, ni_h2h=0.1,)
    print(ei_cell)
    pass
