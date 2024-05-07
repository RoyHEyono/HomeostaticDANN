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
from PIL import Image
from torchvision import transforms as transform
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
    def __init__(self, path, n=0, true_idx=0, remove_digit=None, flatten_bool=True, transforms=None, to_device=True):
        """
        n is the number of distractors, so n=0 is normal mnist 
        """
        # if transforms is not None: raise # not implemented!
         
        super().__init__()

        self.remove_digit = remove_digit
        self.x, self.y = torch.load(path)

        self.transforms = transforms

        
        if self.remove_digit is not None:
            self.indices = [i for i, (img, label) in enumerate(zip(self.x, self.y)) if label not in self.remove_digit]
        
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
        if self.remove_digit is not None:
            return len(self.indices)
        return self.x.size(0)-self.n
    
    def __getitem__(self, idx):
        if self.remove_digit is not None:
            idx = self.indices[idx]
        if self.flatten:
            img = self.x[idx:idx+self.n+1]
            
            if self.transforms:
                img = self.transforms(img)


            if self.idx_permute is not None:
                img, label = img.view(-1)[self.idx_permute], self.y[idx]
            else:
                img, label = img.view(-1), self.y[idx]
        else:
            if self.idx_permute is not None:
                print("Not yet implemented permuation for non flattened data")
                # e.g. [self.idx_permute].view(1, h, w)
            else:
                img, label = self.x[idx:idx+self.n+1], self.y[idx]

        

        return img, label

class ToCudaTransform:
    def __call__(self, x):
        return x.to('cuda')

class ToNumpy:
    def __call__(self, pic):
        return np.array(pic)

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

class SparseFashionMnistDataset(Dataset):
    """
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    Mnist with n+1 digits presented at once, 1st is true target. In this
    implementation the same distractors are presented with each true target. 
    An alternative would be to make them random. 

    Permutate data by setting idx_permute attribute
    https://discuss.pytorch.org/t/permutate-mnist-help-needed/22901
    
    """
    def __init__(self, path, n=0, true_idx=0, remove_digit=None, flatten_bool=True, transforms=None, to_device=True):
        """
        n is the number of distractors, so n=0 is normal mnist 
        """
        # if transforms is not None: raise # not implemented! 
        super().__init__()

        self.transforms = transforms

        self.remove_digit = remove_digit
        self.x, self.y = path
            
        if self.remove_digit is not None:
            self.indices = [i for i, (img, label) in enumerate(zip(self.x, self.y)) if label.argmax(0).item() not in self.remove_digit]
        
        if transforms is None:
            self.x = self.x.float().div(255)
        else:
            self.x = self.transforms(self.x)

        if to_device:
            device = train_utils.get_device()
            self.x = self.x.to(device)
            self.y = self.y.to(device)
        self.n_classes = len(self.y.unique())
        self.n = n # no of distracting
        self.flatten=flatten_bool
        

        self.idx_permute = None

    def __len__(self):
        if self.remove_digit is not None:
            return len(self.indices)
        return self.x.size(0)-self.n
    
    def __getitem__(self, idx):
        if self.remove_digit is not None:
            idx = self.indices[idx]
        if self.flatten:
            img = self.x[idx:idx+self.n+1]

            
            
            if self.idx_permute is not None:
                return img.view(-1)[self.idx_permute], self.y[idx].argmax(0)
            else:
                return img.view(-1), self.y[idx].argmax(0)
        else:
            if self.idx_permute is not None:
                print("Not yet implemented permuation for non flattened data")
                # e.g. [self.idx_permute].view(1, h, w)
            else:
                img = self.x[idx:idx+self.n+1]

                return img, self.y[idx].argmax(0)

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


def get_sparse_kmnist_dataloaders(p, transforms=None, n_kmnist_distractors=0):
    kmnist_path = "/network/datasets/kmnist.var/kmnist_torchvision/KMNIST/processed"
    batch_size = p.train.batch_size
    eval_batch_size = 10000
    if p.train.use_testset:
        train_dataset = SparseMnistDataset(f"{kmnist_path}/training.pt", n_kmnist_distractors, transforms=transforms)
        train_dataset_eval = SparseMnistDataset(f"{kmnist_path}/training.pt", n_kmnist_distractors)
        test_dataset  = SparseMnistDataset(f"{kmnist_path}/test.pt", n_kmnist_distractors)
        
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
        train_dataloader_eval = DataLoader(train_dataset_eval, eval_batch_size, shuffle=True)
        test_dataloader  = DataLoader(test_dataset, eval_batch_size, shuffle=False)
    else:
        val_size = 10000
        dtrain = SparseMnistDataset(f"{kmnist_path}/training.pt", n_kmnist_distractors,transforms=transforms)
        deval  = SparseMnistDataset(f"{kmnist_path}/training.pt", n_kmnist_distractors,transforms=None)
        train_dataloader, train_dataloader_eval, test_dataloader =  get_train_eval_val_dataloaders(
                                                                    dtrain, deval, val_size, batch_size, 
                                                                    eval_batch_size)

    return {"train":train_dataloader, "train_eval":train_dataloader_eval, "test":test_dataloader}


def get_sparse_remove_one_mnist_dataloaders(p, transforms=None, n_mnist_distractors=0, rm_digits=None):
    mnist_path = "/network/datasets/mnist.var/mnist_torchvision/MNIST/processed"
    batch_size = p.train.batch_size
    if p.train.use_testset:
        train_dataset = SparseMnistDataset(f"{mnist_path}/training.pt", n_mnist_distractors, remove_digit=rm_digits)
        train_dataset_eval = SparseMnistDataset(f"{mnist_path}/training.pt", n_mnist_distractors, remove_digit=rm_digits)
        test_dataset  = SparseMnistDataset(f"{mnist_path}/test.pt", n_mnist_distractors, remove_digit=rm_digits, transforms=transforms)
        held_out_digits_dataset = SparseMnistDataset(f"{mnist_path}/test.pt", n_mnist_distractors, remove_digit=[i for i in range(10) if i not in rm_digits])
        
        train_eval_batch_size = len(train_dataloader_eval)
        test_batch_size = len(test_dataset)
        held_out_batch_size = len(held_out_digits_dataset)
        
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
        train_dataloader_eval = DataLoader(train_dataset_eval, train_eval_batch_size, shuffle=True)
        test_dataloader  = DataLoader(test_dataset, test_batch_size, shuffle=False)
        held_out_dataloader = DataLoader(held_out_digits_dataset, held_out_batch_size, shuffle=False)
    else:
        val_size = 10000
        dtrain = SparseMnistDataset(f"{mnist_path}/training.pt", n_mnist_distractors, remove_digit=rm_digits, transforms=transforms)
        deval  = SparseMnistDataset(f"{mnist_path}/training.pt", n_mnist_distractors, remove_digit=rm_digits, transforms=transforms)
        held_out_digits_dataset = SparseMnistDataset(f"{mnist_path}/training.pt", n_mnist_distractors, remove_digit=[i for i in range(10) if i not in rm_digits], transforms=transforms)
        train_eval_batch_size = len(deval)
        held_out_batch_size = len(held_out_digits_dataset)
        train_dataloader, train_dataloader_eval, test_dataloader =  get_train_eval_val_dataloaders(
                                                                    dtrain, deval, val_size, batch_size, 
                                                                    train_eval_batch_size)
        held_out_dataloader = DataLoader(held_out_digits_dataset, held_out_batch_size, shuffle=False)


    return {"train":train_dataloader, "train_eval":train_dataloader_eval, "test":test_dataloader, "ood":held_out_dataloader}


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
    
    # Contrast transform
    contrast_transform = transform.Compose([
                            ContrastStretching(min_percentile=0, max_percentile=100),
                            transform.Lambda(lambda x: x / 255.0),  # Divide by 255
                            ToCudaTransform()
                        ])

    train_img = read_idx_file(f'{fashionmnist_path}/{train_image_path}')
    train_lbl = read_idx_file(f'{fashionmnist_path}/{train_label_path}')
    test_img = read_idx_file(f'{fashionmnist_path}/{test_image_path}')
    test_lbl = read_idx_file(f'{fashionmnist_path}/{test_label_path}')

    training_data = preprocess_data(train_img, train_lbl)
    testing_data = preprocess_data(test_img, test_lbl)

    batch_size = p.train.batch_size
    eval_batch_size = 10000

    if p.train.use_testset:
        train_dataset = SparseFashionMnistDataset(training_data, n_mnist_distractors, transforms=contrast_transform)
        train_dataset_eval = SparseFashionMnistDataset(training_data, n_mnist_distractors, transforms=contrast_transform)
        test_dataset  = SparseFashionMnistDataset(testing_data, n_mnist_distractors, transforms=contrast_transform)
        
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
        train_dataloader_eval = DataLoader(train_dataset_eval, eval_batch_size, shuffle=True)
        test_dataloader  = DataLoader(test_dataset, eval_batch_size, shuffle=False)
    else:
        val_size = 10000
        dtrain = SparseFashionMnistDataset(training_data, n_mnist_distractors,transforms=contrast_transform)
        deval  = SparseFashionMnistDataset(training_data, n_mnist_distractors,transforms=contrast_transform)
        train_dataloader, train_dataloader_eval, test_dataloader =  get_train_eval_val_dataloaders(
                                                                    dtrain, deval, val_size, batch_size, 
                                                                    eval_batch_size)


    return {"train":train_dataloader, "train_eval":train_dataloader_eval, "test":test_dataloader} 

def get_sparse_remove_one_fashionmnist_dataloaders(p, transforms=None, n_mnist_distractors=0, rm_items=None):
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
        train_dataset = SparseFashionMnistDataset(training_data, n_mnist_distractors, remove_digit=rm_items, transforms=transforms)
        train_dataset_eval = SparseFashionMnistDataset(training_data,  n_mnist_distractors, remove_digit=rm_items)
        test_dataset  = SparseFashionMnistDataset(testing_data, n_mnist_distractors, remove_digit=rm_items)
        held_out_digits_dataset = SparseFashionMnistDataset(testing_data, n_mnist_distractors, remove_digit=[i for i in range(10) if i not in rm_items])

        train_eval_batch_size = len(train_dataloader_eval)
        test_batch_size = len(test_dataset)
        held_out_batch_size = len(held_out_digits_dataset)
        
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
        train_dataloader_eval = DataLoader(train_dataset_eval, train_eval_batch_size, shuffle=True)
        test_dataloader  = DataLoader(test_dataset, test_batch_size, shuffle=False)
        held_out_dataloader = DataLoader(held_out_digits_dataset, held_out_batch_size, shuffle=False)
    else:
        val_size = 10000
        dtrain = SparseFashionMnistDataset(training_data, n_mnist_distractors, remove_digit=rm_items, transforms=transforms)
        deval  = SparseFashionMnistDataset(training_data, n_mnist_distractors, remove_digit=rm_items, transforms=None)
        held_out_digits_dataset = SparseFashionMnistDataset(training_data, n_mnist_distractors, remove_digit=[i for i in range(10) if i not in rm_items])

        train_eval_batch_size = len(deval)
        held_out_batch_size = len(held_out_digits_dataset)
        train_dataloader, train_dataloader_eval, test_dataloader =  get_train_eval_val_dataloaders(
                                                                    dtrain, deval, val_size, batch_size, 
                                                                    train_eval_batch_size)

        held_out_dataloader = DataLoader(held_out_digits_dataset, held_out_batch_size, shuffle=False)


    return {"train":train_dataloader, "train_eval":train_dataloader_eval, "test":test_dataloader, "ood":held_out_dataloader}


if __name__ == "__main__":
    get_sparse_fashionmnist_dataloaders()

