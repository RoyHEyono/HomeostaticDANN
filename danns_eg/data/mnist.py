#import cv2
import os
import sys
import numpy as np 

from pathlib import Path
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset
import danns_eg.utils as train_utils
import struct
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms as trnf
from torchvision import datasets
import torch.nn.functional as F
# from ffcv.transforms import ToDevice
# from config import DATASETS_DIR
# import train_utils

class RandomAdjustBrightness:
    def __init__(self, brightness_factor, fixed=False):
        """
        Initializes the RandomAdjustBrightness transform.
        
        Parameters:
        brightness_factor (float): The maximum absolute value by which to adjust the brightness.
        """
        self.brightness_factor = brightness_factor
        self.fixed = fixed

    def __call__(self, x):
        """
        Applies the random brightness adjustment to the input image.

        Parameters:
        x (Tensor): The input image tensor.

        Returns:
        Tensor: The brightness-adjusted image.
        """

        random_adjustment = (np.random.rand() * 2 - 1) * self.brightness_factor

        # Generate a random value within the range [-brightness_factor, brightness_factor]
        if self.fixed:
            random_adjustment = self.brightness_factor
            
        # Adjust the brightness
        x = x + random_adjustment
        # Clip the values to ensure they remain within [0, 1]
        return torch.clamp(x, 0, 1)

# Define custom contrast stretching function
def contrast_stretching(img, min_percentile=0, max_percentile=100):
    #min_val = np.percentile(img.numpy(), min_percentile, axis=None)
    min_val = min_percentile.item()
    #max_val = np.percentile(img.cpu().numpy(), max_percentile, axis=None)
    max_val = 255 # NOTE: Might not be the case
    stretched_img = torch.clamp((img - min_val) * (255.0 / (max_val - min_val)), 0, 255).to(torch.uint8)
    return stretched_img

class ContrastStretching(object):
    def __init__(self, min_percentile=0, max_percentile=100):
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile

    def __call__(self, img):
        # Convert torch tensor to numpy array
        img_np = img

        sample_min = torch.rand(1) * self.min_percentile

        stretched_img = contrast_stretching(img_np, sample_min, self.max_percentile)

        return stretched_img

class ConsistentPermutationInvariantMNISTDataset(Dataset):
    def __init__(self, mnist_dataset, permutation_order):
        self.mnist_dataset = mnist_dataset
        self.permutation_order = permutation_order

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        image, label = self.mnist_dataset[idx]
        permuted_image = image.view(-1)[self.permutation_order].view(image.size())
        return permuted_image, label

def get_sparse_permutation_invariant_mnist_dataloaders(p, permutation_invariant=False):
    # Define transformation to be applied to the data
    transform = trnf.Compose([
        trnf.PILToTensor(), # Convert image to tensor,
        # trnf.Normalize((0.1307,), (0.3081,)),
        trnf.Lambda(lambda x: x / 255.0),
        trnf.Lambda(lambda x: x.view(x.size(0), -1)),
        #transforms.Normalize((0.5,), (0.5,)) # Normalize pixel values to the range [-1, 1]
    ])

    # Download and load the training dataset
    train_dataset = datasets.MNIST(root='/network/datasets/mnist.var/mnist_torchvision/', train=True, transform=transform, download=False)
    # Download and load the test dataset
    test_dataset = datasets.MNIST(root='/network/datasets/mnist.var/mnist_torchvision/', train=False, transform=transform, download=False)

    if permutation_invariant:
        # Generate a fixed permutation order
        permutation_order = torch.randperm(28 * 28)

        # Create permutation invariant datasets with consistent permutation order
        train_dataset = ConsistentPermutationInvariantMNISTDataset(train_dataset, permutation_order)
        test_dataset = ConsistentPermutationInvariantMNISTDataset(test_dataset, permutation_order)

    if p.train.use_testset:
        # Create a dataloader for the training dataset
        train_dataloader = DataLoader(train_dataset, batch_size=p.train.batch_size, shuffle=True)
        
        # Create a dataloader for the test dataset
        test_dataloader = DataLoader(test_dataset, batch_size=p.train.batch_size, shuffle=True)
    else:
        dataset_size = len(train_dataset)
        data_indices = list(range(dataset_size))
        np.random.shuffle(data_indices) # in-place, seed set outside function

        train_idx = data_indices[10000:]
        valid_idx = data_indices[:10000]

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        val_sampler   = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

        # Create a dataloader for the training dataset
        train_dataloader = DataLoader(train_dataset, batch_size=p.train.batch_size, sampler=train_sampler)

        # Create a dataloader for the test dataset
        test_dataloader = DataLoader(train_dataset, batch_size=p.train.batch_size, sampler=val_sampler)

    return {"train":train_dataloader, "test":test_dataloader}


def get_sparse_permutation_invariant_fashionmnist_dataloaders(p=None, permutation_invariant=False, contrast=False, brightness_factor=0, brightness_factor_eval=0):
    # Define transformation to be applied to the data

    if p: 
        batchsize = p.train.batch_size
        use_test = p.train.use_testset
    else: 
        batchsize = 32
        use_test = True

    train_transform = [trnf.PILToTensor(), # Convert image to tensor
                        trnf.Lambda(lambda x: x / 255.0),]
    test_transform = [trnf.PILToTensor(), # Convert image to tensor
                        trnf.Lambda(lambda x: x / 255.0),]

    if contrast:
        train_transform.append(ContrastStretching(min_percentile=0, max_percentile=100))
        test_transform.append(ContrastStretching(min_percentile=0, max_percentile=100))

    train_transform.extend([ 
                RandomAdjustBrightness(brightness_factor),
                trnf.Lambda(lambda x: x.view(x.size(0), -1))
            ])
    test_transform.extend([ 
                RandomAdjustBrightness(brightness_factor_eval if brightness_factor_eval else brightness_factor, fixed=brightness_factor_eval),
                trnf.Lambda(lambda x: x.view(x.size(0), -1))
            ])
    
    train_transform = trnf.Compose(train_transform)
    test_transform = trnf.Compose(test_transform)

    # Download and load the training dataset
    # train_dataset = datasets.FashionMNIST(root="/network/datasets/fashionmnist.var/fashionmnist_torchvision/FashionMNIST/", train=True, transform=train_transform, download=False)
    # test_dataset = datasets.FashionMNIST(root="/network/datasets/fashionmnist.var/fashionmnist_torchvision/FashionMNIST", train=False, transform=test_transform if brightness_factor_eval else train_transform, download=False)
    train_dataset = datasets.FashionMNIST(root="/home/mila/r/roy.eyono/data/FashionMNIST/", train=True, transform=train_transform, download=True)
    test_dataset = datasets.FashionMNIST(root="/home/mila/r/roy.eyono/data/FashionMNIST", train=False, transform=test_transform if brightness_factor_eval else train_transform, download=False)

    

    if permutation_invariant:
        # Generate a fixed permutation order
        permutation_order = torch.randperm(28 * 28)

        # Create permutation invariant datasets with consistent permutation order
        train_dataset = ConsistentPermutationInvariantMNISTDataset(train_dataset, permutation_order)
        test_dataset = ConsistentPermutationInvariantMNISTDataset(test_dataset, permutation_order)

    if use_test or brightness_factor_eval:
        # Create a dataloader for the training dataset
        train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=False, num_workers=2, pin_memory=True )
        
        # Create a dataloader for the test dataset
        test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False, num_workers=2, pin_memory=True )
    else:
        dataset_size = len(train_dataset)
        data_indices = list(range(dataset_size))
        np.random.shuffle(data_indices) # in-place, seed set outside function

        train_idx = data_indices[10000:]
        valid_idx = data_indices[:10000]

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        val_sampler   = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

        # Create a dataloader for the training dataset
        train_dataloader = DataLoader(train_dataset, batch_size=batchsize, sampler=train_sampler)

        # Create a dataloader for the test dataset
        test_dataloader = DataLoader(train_dataset, batch_size=batchsize, sampler=val_sampler)

    return {"train":train_dataloader, "test":test_dataloader}


if __name__ == "__main__":
    get_sparse_permutation_invariant_fashionmnist_dataloaders()

