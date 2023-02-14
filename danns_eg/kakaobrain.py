"""
This model definition comes from https://docs.ffcv.io/ffcv_examples/cifar10.html
    Step 3: Setup model architecture and optimization parameters
    For the model, we use a custom ResNet-9 architecture from KakaoBrain.

https://pytorch.org/blog/tensor-memory-format-matters/
"""

import torch.nn as nn
import torch
import numpy as np

class Mul(nn.Module):
    def __init__(self, weight):
       super(Mul, self).__init__()
       self.weight = weight
    def forward(self, x): return x * self.weight

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x): return x + self.module(x)

def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                         stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
    )
def construct_KakaoBrain_model():
    num_class = 10
    model = nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(2),
        Residual(nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        nn.Linear(128, num_class, bias=False),
        Mul(0.2)
    )
    model = model.to(memory_format=torch.channels_last).cuda()
    return model
