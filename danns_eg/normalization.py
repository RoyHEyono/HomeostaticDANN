# This implementation of GroupNormalization comes from the original paper:
# Figure 3 in https://arxiv.org/pdf/1803.08494.pdf

import torch
import torch.nn as nn

class CustomGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, affine=True, eps=1e-5, momentum=1e-4, subtractive=False, divisive=False):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.running_mean = 0
        self.running_var = 0
        self.momentum = momentum
        self.subtractive = subtractive
        self.divisive = divisive
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        # Reshape the input tensor so that the spatial dimensions and channels are grouped together
        # We assume that the input has shape (batch_size, num_channels, height, width)
        batch_size, num_channels, height, width = x.size()
        x = x.view(batch_size, self.num_groups, num_channels // self.num_groups, height, width)

        mean = torch.mean(x, dim=(2,3,4), keepdim=True)
        var = torch.var(x, dim=(2,3,4), unbiased=False, keepdim=True)
        # Apply normalization
        if self.subtractive and not self.divisive:
            x = (x - mean)
        elif self.divisive and not self.subtractive:
            x = x / torch.sqrt(var + self.eps)
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        
        # Reshape the normalized tensor back to its original shape
        x = x.view(batch_size, num_channels, height, width)
        
        # Apply the learned weight and bias
        if self.affine:
            x = x * self.weight + self.bias
        
        return x

# Karparthy Implementation of LayerNorm: https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.py
class LayerNormKarpathy:

    @staticmethod
    def forward(x, w, b):
        eps = 1e-5
        B, T, C = x.size()
        mean = x.sum(-1, keepdim=True) / C # B,T,1
        xshift = x - mean # B,T,C
        var = (xshift**2).sum(-1, keepdim=True) / C # B,T,1
        rstd = (var + eps) ** -0.5 # B,T,1
        norm = xshift * rstd # B,T,C
        out = norm * w + b # B,T,C

        cache = (x, w, mean, rstd)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        x, w, mean, rstd = cache
        # recompute the norm (save memory at the cost of compute)
        norm = (x - mean) * rstd
        # gradients for weights, bias
        db = dout.sum((0, 1))
        dw = (dout * norm).sum((0, 1))
        # gradients for input
        dnorm = dout * w
        dx = dnorm - dnorm.mean(-1, keepdim=True) - norm * (dnorm * norm).mean(-1, keepdim=True)
        dx *= rstd
        return dx, dw, db

class LayerNormalize(nn.Module):
    def __init__(self, feature_size, eps=1e-5):
        super(LayerNormalize, self).__init__()
        self.eps = eps  # Small value to prevent division by zero

    def forward(self, x):
        # Calculate mean and variance
        with torch.no_grad():
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize the input
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        return x_norm

class MeanNormalizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, no_backward, no_forward=False):

        if no_forward:
            ctx.no_backward = no_backward
            return x

        mean = x.mean(dim=-1, keepdim=True)
        x_norm = x - mean
        ctx.save_for_backward(mean)
        ctx.no_backward = no_backward
        return x_norm

    @staticmethod
    def backward(ctx, grad_output):
        # mean, = ctx.saved_tensors
        # N = grad_output.shape[-1]

        if ctx.no_backward:
            return grad_output, None, None  # Second output corresponds to `no_backward`, which has no gradient

        grad_mean = grad_output.mean(dim=-1, keepdim=True)
        grad_x = grad_output - grad_mean

        return grad_x, None, None  # Second `None` is for `no_backward`, which is not trainable

class MeanNormalize(nn.Module):
    def __init__(self, no_backward=False, no_forward=False):
        super(MeanNormalize, self).__init__()
        self.no_backward = no_backward
        self.no_forward = no_forward

    def forward(self, x):
        return MeanNormalizeFunction.apply(x, self.no_backward, self.no_forward)


class DivisiveNormalizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, no_backward, no_forward=False):

        if no_forward:
            ctx.no_backward = no_backward
            return input
        
        # Compute variance along the last dimension (no mean subtraction, unbiased=False for population variance)
        var = input.var(dim=-1, keepdim=True, unbiased=False)
        # Compute sigma = sqrt(var + epsilon) with epsilon for numerical stability
        epsilon = 1e-5
        sigma = torch.sqrt(var + epsilon)
        # Divisive normalization (no centering)
        output = input / sigma
        # Save tensors for backward computations
        ctx.save_for_backward(input, sigma)
        ctx.eps = epsilon  # store epsilon if needed
        ctx.no_backward = no_backward
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):

        if ctx.no_backward:
            return grad_output, None, None

        # Retrieve saved tensors
        input, sigma = ctx.saved_tensors
        # Number of features along the normalized dimension
        N = input.size(-1)
        # Compute mean of input along last dim (for gradient formula, though mean was not used in forward output)
        mu = input.mean(dim=-1, keepdim=True)
        # Compute dot product of grad_output and input along last dimension (sum of elementwise products)
        grad_out_dot_x = (grad_output * input).sum(dim=-1, keepdim=True)
        # Gradient for the input (applying the derived formula)
        grad_input = grad_output / sigma  -  (input - mu) * grad_out_dot_x / (sigma**3 * N)
        return grad_input, None, None


class DivisiveNormalize(nn.Module):
    def __init__(self, no_backward=False, no_forward=False):
        super(DivisiveNormalize, self).__init__()
        self.no_backward = no_backward
        self.no_forward = no_forward

    def forward(self, x):
        return DivisiveNormalizeFunction.apply(x, self.no_backward, self.no_forward)


class LayerNormalizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, no_backward, no_forward=False):

        
        if no_forward:
            epsilon = 1e-5
            # Compute mean and variance along last dimension
            mu = x.mean(dim=-1, keepdim=True)
            # var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            sigma = torch.sqrt(var + epsilon)               # standard deviation
            ctx.no_backward = no_backward
            ctx.save_for_backward(x, mu, sigma)
            return x
        
        epsilon = 1e-5
        # Compute mean and variance along last dimension
        mu = x.mean(dim=-1, keepdim=True)
        # var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        sigma = torch.sqrt(var + epsilon)               # standard deviation
        y = (x-mu) / sigma                                   # normalized output
        # Save tensors needed for backward
        
        ctx.save_for_backward(x, mu, sigma)
        ctx.no_backward = no_backward
        return y

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.no_backward:
            return grad_output, None, None

        x, mu, sigma= ctx.saved_tensors
        y = (x-mu) / sigma
        D = x.shape[-1]

        grad_mean = grad_output.mean(dim=-1, keepdim=True)
        dot = (grad_output * y).sum(dim=-1, keepdim=True)

        grad_input = (grad_output - grad_mean - y * dot / D) / sigma
        return grad_input, None, None

class LayerNormalizeCustom(nn.Module):
    def __init__(self, no_backward=False, no_forward=False):
        super(LayerNormalizeCustom, self).__init__()
        self.no_backward = no_backward
        self.no_forward = no_forward

    def forward(self, x):
        return LayerNormalizeFunction.apply(x, self.no_backward, self.no_forward)

class LayerNormalizeFunctionFA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, var, weights, no_backward):
        
        epsilon = 1e-5
        ctx.no_backward = no_backward
        ctx.save_for_backward(x, var, weights.to(x.device))
        ctx.actual_var = x.var(dim=-1, keepdim=True, unbiased=False)
        ctx.actual_var = torch.sqrt(ctx.actual_var + epsilon) 
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.no_backward:
            return grad_output, None, None, None, None

        #TODO: Need to add some flags to better manage runs and configs
        x, var, weights = ctx.saved_tensors
        D = x.shape[-1]

        # grad_mean = (weights * grad_output).sum(dim=-1, keepdim=True)
        grad_mean = grad_output.mean(dim=-1, keepdim=True)
        dot = (grad_output * x).sum(dim=-1, keepdim=True)

        grad_input = (grad_output - grad_mean- x * dot / D) / ctx.actual_var
        # grad_input = (grad_output - grad_mean) / ctx.actual_var
        # grad_input = (grad_output) / ctx.actual_var
        # grad_input = (grad_output - grad_mean)
        # grad_input = (grad_output - x * dot / D)
        return grad_input, None, None, None, None

class LayerNormalizeCustomFA(nn.Module):
    def __init__(self, weights, no_backward=False):
        super().__init__()
        self.no_backward = no_backward
        self.weights = weights

    def forward(self, x, var):
        return LayerNormalizeFunctionFA.apply(
            x, var, self.weights, self.no_backward
        )