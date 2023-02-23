import cv2
import os
import sys
import numpy as np 

from pathlib import Path
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset

# from config import DATASETS_DIR
# import train_utils

class SparseMnistDataset(Dataset):
    """
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    Mnist with n+1 digits presented at once, 1st is true target. In this
    implementation the same distractors are presented with each true target. 
    An alternative would be to make them random. 

    Permutate data by setting idx_permute attribute
    https://discuss.pytorch.org/t/permutate-mnist-help-needed/22901
    
    """
    def __init__(self, path, n=0, true_idx=0, flatten_bool=True, transforms=None, to_device=True):
        """
        n is the number of distractors, so n=0 is normal mnist 
        """
        if transforms is not None: raise # not implemented! 
        super().__init__()
        self.x, self.y = torch.load(path)
        self.x = self.x.float().div(255)
        self.n_classes = len(self.y.unique())
        self.n = n # no of distracting
        self.flatten=flatten_bool
        if to_device:
            device = train_utils.get_device()
            self.x = self.x.to(device)
            self.y = self.y.to(device)

        self.idx_permute = None

    def __len__(self):
        return self.x.size(0)-self.n
    
    def __getitem__(self, idx):
        if self.flatten:
            if self.idx_permute is not None:
                return self.x[idx:idx+self.n+1].view(-1)[self.idx_permute], self.y[idx]
            else:
                return self.x[idx:idx+self.n+1].view(-1), self.y[idx]
        else:
            if self.idx_permute is not None:
                print("Not yet implemented permuation for non flattened data")
                # e.g. [self.idx_permute].view(1, h, w)
            else:
                return self.x[idx:idx+self.n+1], self.y[idx]

def get_train_eval_val_dataloaders(training_dataset, dataset, val_size, batch_size, val_batch_size=None,
                                   num_workers=0, pin_memory=False):
    """
    training_dataset : dataset with augmentations
    dataset : same as training_dataset but without augmentations
    """
    assert val_size > 0
    if val_batch_size is None:
        val_batch_size = batch_size
    assert len(training_dataset) == len(dataset)

    dataset_size = len(dataset)
    data_indices = list(range(dataset_size))
    np.random.shuffle(data_indices) # in-place, seed set outside function

    train_idx = data_indices[val_size:]
    valid_idx = data_indices[:val_size]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    train_eval_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    val_sampler   = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(training_dataset,
                                               batch_size,
                                               sampler=train_sampler,
                                               num_workers=num_workers,
                                               pin_memory=pin_memory)

    train_eval_loader = torch.utils.data.DataLoader(dataset,
                                               val_batch_size,
                                               sampler=train_eval_sampler,
                                               num_workers=num_workers,
                                               pin_memory=pin_memory)

    val_loader = torch.utils.data.DataLoader(dataset,
                                             val_batch_size,
                                             sampler=val_sampler,
                                             num_workers=num_workers,
                                             pin_memory=pin_memory)

    return train_loader, train_eval_loader, val_loader


def get_sparse_mnist_dataloaders(p, transforms=None, n_mnist_distractors=0):
    mnist_path = DATASETS_DIR/"torchvision/MNIST/processed"
    batch_size = p.train.batch_size
    eval_batch_size = 10000
    if p.train.use_testset:
        train_dataset = SparseMnistDataset(mnist_path/"training.pt", n_mnist_distractors, transforms=transforms)
        train_dataset_eval = SparseMnistDataset(mnist_path/"training.pt", n_mnist_distractors)
        test_dataset  = SparseMnistDataset(mnist_path/"test.pt", n_mnist_distractors)
        
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
        train_dataloader_eval = DataLoader(train_dataset_eval, eval_batch_size, shuffle=True)
        test_dataloader  = DataLoader(test_dataset, eval_batch_size, shuffle=False)
    else:
        val_size = 10000
        dtrain = SparseMnistDataset(mnist_path/"training.pt", n_mnist_distractors,transforms=transforms)
        deval  = SparseMnistDataset(mnist_path/"training.pt", n_mnist_distractors,transforms=None)
        train_dataloader, train_dataloader_eval, test_dataloader =  get_train_eval_val_dataloaders(
                                                                    dtrain, deval, val_size, batch_size, 
                                                                    eval_batch_size)

    return {"train":train_dataloader, "train_eval":train_dataloader_eval, "test":test_dataloader} 

