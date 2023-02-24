"""
Resnet implementations. 

See https://myrtle.ai/learn/how-to-train-your-resnet/
https://docs.ffcv.io/ffcv_examples/cifar10.html

Note: All Ei models are hardcoded to have 10% inhibitory in this module. 
"""
import torch.nn as nn
import torch
import numpy as np 

from danns_eg.conv import EiConvLayer, ConvLayer
from danns_eg.dense import EiDenseLayer
from danns_eg.sequential import Sequential

class Mul(nn.Module):
    def __init__(self, weight):
       super(Mul, self).__init__()
       self.weight = weight
    def forward(self, x): return x * self.weight

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

def conv(p, c_in, c_out, kernel_size=3, stride=1, padding=1, groups=1):
    modules = []
    if p.model.is_dann == True:
        conv2d = EiConvLayer(c_in, c_out, int(c_out*0.1),kernel_size,kernel_size,
                            stride=stride,padding=padding, groups=groups, bias=False)
    else:
        conv2d = ConvLayer(c_in, c_out, kernel_size=kernel_size,stride=stride,
                           padding=padding, groups=groups, bias=False)
    modules.append(conv2d)
    
    if p.model.normtype == "bn": norm_layer = nn.BatchNorm2d(c_out)
    elif p.model.normtype == "ln": norm_layer = nn.GroupNorm(1,c_out)
    elif p.model.normtype.lower() == "none": norm_layer = None
    if norm_layer is not None: modules.append(norm_layer)
    # then everything will use relu for now
    act_func = nn.ReLU(inplace=True)
    modules.append(act_func)

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

def resnet9_kakaobrain(p:dict, linear_decoder=False):
    """
    From the ffcv example
    # Model (from KakaoBrain: https://github.com/wbaek/torchskeleton)
    # https://github.com/libffcv/ffcv/blob/main/examples/cifar/train_cifar.py
    # For the model, we use a custom ResNet-9 architecture from KakaoBrain.
    # https://docs.ffcv.io/ffcv_examples/cifar10.html

    Also see:
        https://pytorch.org/blog/tensor-memory-format-matters/
    
    Args:
        p: parameter config object 
        linear_decoder: if True, the model's final layer is always linear.
    """
    num_class = 10
    
    modules = [
        conv(p, 3, 64, kernel_size=3, stride=1, padding=1),
        conv(p, 64, 128, kernel_size=5, stride=2, padding=2),
        BasicBlock(conv(p,128, 128), conv(p,128, 128)),
        conv(p,128, 256, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(2),
        BasicBlock(conv(p,256, 256), conv(p,256, 256)),
        conv(p,256, 128, kernel_size=3, stride=1, padding=0),
        nn.AdaptiveMaxPool2d((1, 1)),
        Flatten()]
    
    if linear_decoder or not p.model.is_dann:
        modules.append(nn.Linear(128, num_class, bias=True))
    
    elif p.model.is_dann:
        ni = max(1,int(num_class*0.1))
        modules.append(EiDenseLayer(128, num_class, ni=ni, split_bias=False, # true is EG
                                     use_bias=True))
    
    modules.append(Mul(0.2))
    model = Sequential(modules).to(memory_format=torch.channels_last).cuda()
    return model