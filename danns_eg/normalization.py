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
    def forward(ctx, x, no_backward, no_forward=False):

        if no_forward:
            ctx.no_backward = no_backward
            return x

        epsilon = 1e-5
        # Compute mean and variance along last dimension
        mu = x.mean(dim=-1, keepdim=True)
        # var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        sigma = torch.sqrt(var + epsilon)               # standard deviation
        y = x / sigma                                   # normalized output
        # Save tensors needed for backward
        ctx.save_for_backward(x, mu, sigma)
        ctx.no_backward = no_backward
        return y

    @staticmethod
    def backward(ctx, grad_output):

        if ctx.no_backward:
            return grad_output, None, None

        x, mu, sigma = ctx.saved_tensors
        D = x.shape[-1]                                 # size of last dimension
        # Compute dot = sum_k (grad_out[k] * x[k]) along last dimension
        dot = (grad_output * x).sum(dim=-1, keepdim=True)
        # Apply the derived gradient formula
        grad_input = grad_output / sigma - (x - mu) * (dot / D) / (sigma ** 3)
        return grad_input, None, None

class DivisiveNormalize(nn.Module):
    def __init__(self, no_backward=False, no_forward=False):
        super(DivisiveNormalize, self).__init__()
        self.no_backward = no_backward
        self.no_forward = no_forward

    def forward(self, x):
        return DivisiveNormalizeFunction.apply(x, self.no_backward, self.no_forward)

# main function
if __name__ == "__main__":
    x = torch.randn(32, 3, 28, 28) # batch_size, num_channels, height, width
    norm1 = nn.GroupNorm(1, 3, affine=False, eps=1e-5)
    norm2 = SubtractiveOnlyGroupNorm(1, 3, affine=False, eps=1e-5)
    y1 = norm1(x)[3][0][0][1]
    y2 = norm2(x)[3][0][0][1]
    print(y1, y2)
    print((norm1(x) - norm2(x)).abs().max())