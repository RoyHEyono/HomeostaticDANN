import cv2
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
        if isinstance(path, str):
            self.x, self.y = torch.load(path)
        else:
            self.x, self.y = path
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
    mnist_path = "/network/datasets/mnist.var/mnist_torchvision/MNIST/processed"
    batch_size = p.train.batch_size
    eval_batch_size = 10000
    if p.train.use_testset:
        train_dataset = SparseMnistDataset(f"{mnist_path}/training.pt", n_mnist_distractors, transforms=transforms)
        train_dataset_eval = SparseMnistDataset(f"{mnist_path}/training.pt", n_mnist_distractors)
        test_dataset  = SparseMnistDataset(f"{mnist_path}/test.pt", n_mnist_distractors)
        
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
        train_dataloader_eval = DataLoader(train_dataset_eval, eval_batch_size, shuffle=True)
        test_dataloader  = DataLoader(test_dataset, eval_batch_size, shuffle=False)
    else:
        val_size = 10000
        dtrain = SparseMnistDataset(f"{mnist_path}/training.pt", n_mnist_distractors,transforms=transforms)
        deval  = SparseMnistDataset(f"{mnist_path}/training.pt", n_mnist_distractors,transforms=None)
        train_dataloader, train_dataloader_eval, test_dataloader =  get_train_eval_val_dataloaders(
                                                                    dtrain, deval, val_size, batch_size, 
                                                                    eval_batch_size)

    return {"train":train_dataloader, "train_eval":train_dataloader_eval, "test":test_dataloader}


def get_sparse_remove_one_mnist_dataloaders(p, transforms=None, n_mnist_distractors=0):
    mnist_path = "/network/datasets/mnist.var/mnist_torchvision/MNIST/processed"
    batch_size = p.train.batch_size
    eval_batch_size = 10000
    if p.train.use_testset:
        train_dataset = SparseMnistDataset(f"{mnist_path}/training.pt", n_mnist_distractors, transforms=transforms)
        train_dataset_eval = SparseMnistDataset(f"{mnist_path}/training.pt", n_mnist_distractors)
        test_dataset  = SparseMnistDataset(f"{mnist_path}/test.pt", n_mnist_distractors)
        
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
        train_dataloader_eval = DataLoader(train_dataset_eval, eval_batch_size, shuffle=True)
        test_dataloader  = DataLoader(test_dataset, eval_batch_size, shuffle=False)
    else:
        val_size = 10000
        dtrain = SparseMnistDataset(f"{mnist_path}/training.pt", n_mnist_distractors,transforms=transforms)
        deval  = SparseMnistDataset(f"{mnist_path}/training.pt", n_mnist_distractors,transforms=None)
        train_dataloader, train_dataloader_eval, test_dataloader =  get_train_eval_val_dataloaders(
                                                                    dtrain, deval, val_size, batch_size, 
                                                                    eval_batch_size)

    return {"train":train_dataloader, "train_eval":train_dataloader_eval, "test":test_dataloader}


def read_idx_file(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def preprocess_data(images, labels):
    # Normalize pixel values to the range [0, 1]
    images = images.astype('float32')
    # Reshape the images if needed
    images = images.reshape(images.shape[0], 28, 28, 1)
    images = torch.tensor(images, dtype=torch.float64)
    # Convert labels to one-hot encoding
    labels = torch.tensor(labels, dtype=torch.int64)
    labels  = torch.nn.functional.one_hot(labels, 10)

    return images, labels

def split_dataset(images, labels, test_size=0.2, random_state=42):
    return train_test_split(images, labels, test_size=test_size, random_state=random_state)


def get_sparse_fashionmnist_dataloaders(p, transforms=None, n_mnist_distractors=0):
    fashionmnist_path = "/network/datasets/fashionmnist.var/fashionmnist_torchvision/FashionMNIST/raw"
    train_image_path = "train-images-idx3-ubyte"
    train_label_path = "train-labels-idx1-ubyte"
    test_image_path = "t10k-images-idx3-ubyte"
    test_label_path = "t10k-labels-idx1-ubyte"

    train_img = read_idx_file(f'{fashionmnist_path}/{train_image_path}')
    train_lbl = read_idx_file(f'{fashionmnist_path}/{train_label_path}')
    test_img = read_idx_file(f'{fashionmnist_path}/{test_image_path}')
    test_lbl = read_idx_file(f'{fashionmnist_path}/{test_label_path}')

    training_data = preprocess_data(train_img, train_lbl)
    testing_data = preprocess_data(test_img, test_lbl)

    batch_size = p.train.batch_size
    eval_batch_size = 10000

    if p.train.use_testset:
        train_dataset = SparseMnistDataset(training_data, n_mnist_distractors, transforms=transforms)
        train_dataset_eval = SparseMnistDataset(training_data, n_mnist_distractors)
        test_dataset  = SparseMnistDataset(testing_data, n_mnist_distractors)
        
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
        train_dataloader_eval = DataLoader(train_dataset_eval, eval_batch_size, shuffle=True)
        test_dataloader  = DataLoader(test_dataset, eval_batch_size, shuffle=False)
    else:
        val_size = 10000
        dtrain = SparseMnistDataset(training_data, n_mnist_distractors,transforms=transforms)
        deval  = SparseMnistDataset(training_data, n_mnist_distractors,transforms=None)
        train_dataloader, train_dataloader_eval, test_dataloader =  get_train_eval_val_dataloaders(
                                                                    dtrain, deval, val_size, batch_size, 
                                                                    eval_batch_size)


    return {"train":train_dataloader, "train_eval":train_dataloader_eval, "test":test_dataloader} 


if __name__ == "__main__":
    get_sparse_fashionmnist_dataloaders()

