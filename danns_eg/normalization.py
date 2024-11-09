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



# main function
if __name__ == "__main__":
    x = torch.randn(32, 3, 28, 28) # batch_size, num_channels, height, width
    norm1 = nn.GroupNorm(1, 3, affine=False, eps=1e-5)
    norm2 = SubtractiveOnlyGroupNorm(1, 3, affine=False, eps=1e-5)
    y1 = norm1(x)[3][0][0][1]
    y2 = norm2(x)[3][0][0][1]
    print(y1, y2)
    print((norm1(x) - norm2(x)).abs().max())