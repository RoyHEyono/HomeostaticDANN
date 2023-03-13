# This implementation of GroupNormalization comes from the original paper:
# Figure 3 in https://arxiv.org/pdf/1803.08494.pdf

import torch
import torch.nn as nn

class SubtractiveOnlyGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, affine=True, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        # Reshape the input tensor so that the spatial dimensions and channels are grouped together
        # We assume that the input has shape (batch_size, num_channels, height, width)
        batch_size, num_channels, height, width = x.size()
        x = x.view(batch_size, self.num_groups, num_channels // self.num_groups, height, width)
        
        # Calculate the mean and variance for each group
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        var = x.var(dim=(2, 3, 4), keepdim=True)
        
        # Apply normalization
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        # Reshape the normalized tensor back to its original shape
        x = x.view(batch_size, num_channels, height, width)
        
        # Apply the learned weight and bias
        if self.affine:
            x = x * self.weight + self.bias
        
        return x


# main function
if __name__ == "__main__":
    x = torch.randn(32, 3, 28, 28) # batch_size, num_channels, height, width
    norm1 = nn.GroupNorm(1, 3, affine=False, eps=1e-5)
    norm2 = SubtractiveOnlyGroupNorm(1, 3, affine=False, eps=1e-5)
    print(norm1(x)[3][0][0][1])
    print(norm2(x)[3][0][0][1])
    print((norm1(x) == norm2(x)).all())