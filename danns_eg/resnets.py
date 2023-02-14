"""
Resnet implementations including ei

https://myrtle.ai/learn/how-to-train-your-resnet/
"""
from typing import Any, Callable, List, Optional, Type, Union

import torch.nn as nn
import torch
import numpy as np 

from lib.conv_layers import EiConvLayer, ConvLayer
from models.sequential import Sequential
#from torch.nn import Sequential 

class Mul(nn.Module):
    def __init__(self, weight):
       super(Mul, self).__init__()
       self.weight = weight
    def forward(self, x): return x * self.weight
class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

def conv(p, c_in, c_out, kernel_size=3, stride=1, padding=1, groups=1):
    if p.model.is_dann == True:
        conv2d = EiConvLayer(c_in, c_out, int(c_out*0.1),kernel_size,kernel_size,
                            stride=stride,padding=padding, groups=groups, bias=False)
        # here also take into account the norm layers and subtractive only 
        # bias will be false unless you are learning the norm!
        
    else:
        # conv2d = nn.Conv2d(c_in, c_out, kernel_size=kernel_size,stride=stride,
        #                    padding=padding, groups=groups, bias=False)
        # nn.init.kaiming_normal_(conv2d.weight, mode="fan_out", nonlinearity="relu")
        conv2d = ConvLayer(c_in, c_out, kernel_size=kernel_size,stride=stride,
                           padding=padding, groups=groups, bias=False)

    if p.model.normtype == "bn": norm_layer = nn.BatchNorm2d(c_out)
    elif p.model.normtype == "ln": norm_layer = nn.GroupNorm(1,c_out)
    elif p.model.normtype.lower() == "none": norm_layer = None

    # then everything will use relu for now
    act_func = nn.ReLU(inplace=True)

    modules = [m for m in [conv2d, norm_layer, act_func] if m is not None]
    return Sequential(modules)

class BasicBlock(nn.Module):
    def __init__(self, layer1, layer2, ds_conv=None):
        super().__init__()
        self.module = Sequential([layer1, layer2]) # this might need to change
        self.downsample_conv = ds_conv
    def forward(self, x):
        identity = x
        if self.downsample_conv is not None:
            identity = self.downsample_conv(x) 
        return identity + self.module(x)

class Bottleneck(nn.Module):
    def __init__(self, ):
        pass

def resnet18(p, cifar=True):
    num_class = 10
    if cifar: conv1 = conv(p, 3, 64,  stride=1)
    else: raise
    conv2_x = BasicBlock(conv(p, 64, 64,   stride=1),
                         conv(p, 64, 64,   stride=1))
    conv3_x = BasicBlock(conv(p, 64, 128,  stride=2),
                         conv(p, 128, 128, stride=1),
                         nn.Conv2d(64, 128, 1, stride=2, padding=0, bias=False))
    conv4_x = BasicBlock(conv(p, 128, 256, stride=2),
                         conv(p, 256, 256, stride=1),
                         nn.Conv2d(128, 256, 1, stride=2, padding=0, bias=False))
    conv5_x = BasicBlock(conv(p, 256, 512, stride=2),
                         conv(p, 512, 512, stride=1),
                         nn.Conv2d(256, 512, 1, stride=2, padding=0, bias=False))

    final = Sequential([
            nn.AdaptiveMaxPool2d((1, 1)),
            Flatten(),
            nn.Linear(512, num_class, bias=False), 
            ])
    
    model = Sequential([conv1, conv2_x, conv3_x, conv4_x, conv5_x, final])
    model = model.to(memory_format=torch.channels_last).cuda()
    #model = model.cuda()
    return model

def resnet9_kakaobrain(p):
    """
    From the ffcv example
    # Model (from KakaoBrain: https://github.com/wbaek/torchskeleton)
    # https://github.com/libffcv/ffcv/blob/main/examples/cifar/train_cifar.py
    # For the model, we use a custom ResNet-9 architecture from KakaoBrain.
    # https://docs.ffcv.io/ffcv_examples/cifar10.html
    
    Args:
        p is the params object 
    """
    num_class = 10
    model = Sequential([
        conv(p, 3, 64, kernel_size=3, stride=1, padding=1),
        conv(p, 64, 128, kernel_size=5, stride=2, padding=2),
        BasicBlock(conv(p,128, 128), conv(p,128, 128)),
        conv(p,128, 256, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(2),
        BasicBlock(conv(p,256, 256), conv(p,256, 256)),
        conv(p,256, 128, kernel_size=3, stride=1, padding=0),
        nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        nn.Linear(128, num_class, bias=False),
        Mul(0.2)]
    )
    model = model.to(memory_format=torch.channels_last).cuda()
    return model