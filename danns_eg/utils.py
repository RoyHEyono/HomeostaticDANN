import shutil
import os
import sys
import random
import yaml
import numpy as np
import dataclasses
import argparse
import time
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.nn as nn 

from pprint import pprint
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
import numpy as np
#from danns_eg.data.mnist import SparseMnistDataset
from torch.utils.data import DataLoader, Dataset

from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.decomposition import IncrementalPCA
import gc
import pickle as pkl
import math

def set_seed_all(seed):
    """
    Sets all random states
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def set_cudnn_flags():
    """Set CuDNN flags for reproducibility"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """Returns torch.device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def get_cpus_on_node() -> int:
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    return cpu_count()

def checkpoint_model(save_path, model, epoch_i, opt, scheduler=None):
    checkpoint_dict = { 'epoch_i': epoch_i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict() }
    if scheduler is not None:
        checkpoint_dict['model_state_dict'] = scheduler.state_dict()
    torch.save(checkpoint_dict, save_path)

def get_config():
    import fastargs
    config = fastargs.get_current_config()
    if not hasattr(sys, 'ps1'): # if not interactive mode 
        parser = argparse.ArgumentParser(description='fastargs demo')
        config.augment_argparse(parser) # adds cl flags and --config-file field to argparser
        config.collect_argparse_args(parser)
        config.validate(mode='stderr')
        config.summary()  # print summary
    return config.get()

def load_config(filepath):
    import fastargs
    config = fastargs.get_current_config()
    config.collect_config_file(filepath)
    config.validate(mode='stderr')
    return config.get()

def get_params_to_log_wandb(p):
    """
    Returns dictionary of parameter configurations we log to wanbd.
    Everything but the "exp" field is logged
    """
    params_to_log = dict()#use_autocast = p.exp.use_autocast)
    for key, field in p.__dict__.items():
        if key == "exp": continue
        else: params_to_log.update(field.__dict__)
    return params_to_log


def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    return mnist.data.to_numpy().astype('float32') / 255.0, mnist.target.to_numpy().astype('int')

def load_fashionmnist():
    fashionmnist = fetch_openml('Fashion-MNIST', version=1, cache=True)
    return fashionmnist.data.to_numpy().astype('float32') / 255.0, fashionmnist.target.to_numpy().astype('int')

def linear_annealing(epoch, T_max):
    return epoch / T_max

def exponential_annealing(epoch, tau):
    return 1 - math.exp(-epoch / tau)

def logarithmic_annealing(epoch, T_max):
    return math.log(1 + epoch) / math.log(1 + T_max)

def cosine_annealing(epoch, T_max):
    return 0.5 * (1 - math.cos(math.pi * epoch / T_max))

# print("Loading data")
# X, y = load_mnist()
# print("Loaded data")


def comparative_mean_plot(layer_output_1, layer_output_2, label1="Label1", label2="Label2"):
    # Create a 1x3 subplot layout
    plt.figure(figsize=(15, 5))  # Adjust the figure size as needed

    for layer_num in range(len(layer_output_1)):
        # Plot the first subplot
        plt.subplot(1, 3, layer_num+1)
        # Plot the mean line
        layer_output = layer_output_1[layer_num]
        mu_mu = layer_output[:,0]
        mu_std = layer_output[:,1]
        x = range(len(mu_mu))
        plt.plot(x, mu_mu, label=label1)
        #plt.fill_between(x, mu_mu - mu_std, mu_mu + mu_std, alpha=0.2, color="green")

        # Plot the mean line
        layer_output = layer_output_2[layer_num]
        mu_mu = layer_output[:,0]
        mu_std = layer_output[:,1]
        x = range(len(mu_mu))
        plt.plot(x, mu_mu, label=label2, alpha=0.2)
        #plt.fill_between(x, mu_mu - mu_std, mu_mu + mu_std, alpha=0.2, color="red")

        # Add labels and title
        plt.xlabel('Steps')
        plt.ylabel('Mean')
        plt.legend()
        plt.title(f'Layer {layer_num}')

    plt.suptitle("Mean")
    
def comparative_var_plot(layer_output_1, layer_output_2, label1="Label1", label2="Label2"):
    # Create a 1x3 subplot layout
    plt.figure(figsize=(15, 5))  # Adjust the figure size as needed

    for layer_num in range(len(layer_output_1)):
        # Plot the first subplot
        plt.subplot(1, 3, layer_num+1)
        # Plot the mean line
        layer_output = layer_output_1[layer_num]
        var_mu = layer_output[:,2]
        var_std = layer_output[:,3]
        x = range(len(var_mu))
        plt.plot(x, var_mu, label=label1)
        #plt.fill_between(x, var_mu - var_std, var_mu + var_std, alpha=0.2, color="green")

        # Plot the mean line
        layer_output = layer_output_2[layer_num]
        var_mu = layer_output[:,2]
        var_std = layer_output[:,3]
        x = range(len(var_mu))
        plt.plot(x, var_mu, label=label2, alpha=0.2)
        #plt.fill_between(x, var_mu - var_std, var_mu + var_std, alpha=0.2, color="red")

        # Add labels and title
        plt.xlabel('Steps')
        plt.ylabel('Var')
        plt.yscale('log')
        plt.legend()
        plt.title(f'Layer {layer_num}')

    plt.suptitle("Variance")


def mean_plot(model, title="mean_line_training"):
    # Create a 1x3 subplot layout
    plt.figure(figsize=(15, 5))  # Adjust the figure size as needed

    # Plot the first subplot
    plt.subplot(1, 3, 1)
    # Plot the mean line
    layer_output = np.array(model.fc1_output)
    mu_mu = layer_output[:,0]
    mu_std = layer_output[:,1]
    x = range(len(mu_mu))
    plt.plot(x, mu_mu, label='Mean Line')
    plt.fill_between(x, mu_mu - mu_std, mu_mu + mu_std, alpha=0.2, label='Standard Deviation')
    # Add labels and title
    plt.xlabel('Steps')
    plt.ylabel('Mean')
    plt.title('Layer 1')

    plt.savefig(f'{title}.pdf')

def var_plot(model, title="var_line_training"):
    # Create a 1x3 subplot layout
    plt.figure(figsize=(15, 5))  # Adjust the figure size as needed

    # Plot the first subplot
    plt.subplot(1, 3, 1)
    # Plot the mean line
    layer_output = np.array(model.fc1_output)
    var_mu = layer_output[:,2]
    var_std = layer_output[:,3]
    x = range(len(var_mu))
    plt.plot(x, var_mu, label='Mean Line')
    plt.fill_between(x, var_mu - var_std, var_mu + var_std, alpha=0.2, label='Standard Deviation')
    plt.yscale('log')
    # Add labels and title
    plt.xlabel('Steps')
    plt.ylabel('Variance')
    plt.title('Layer 1')

    plt.savefig(f'{title}.pdf')


def img_stat(ax):
    X_mnist, _ = load_mnist()
    #X_fashionmnist, _ = load_fashionmnist()

    mu_mnist = []
    var_mnist = []
    mu_fmnist = []
    var_fmnist = []

    for mnist, mnist_translated in zip(X_mnist, X_mnist):
        
        mu_mnist.append(np.mean(mnist))
        var_mnist.append(np.var(mnist))
        mnist_translated = mnist_translated + np.random.normal(loc=0.3, scale=0.1, size=(784))
        mu_fmnist.append(np.mean(mnist_translated))
        var_fmnist.append(np.var(mnist_translated))
        
    
    ax.scatter(mu_fmnist, var_fmnist, c='red', label="MNIST-shifted",
    alpha=0.1)
    ax.scatter(mu_mnist, var_mnist, c='black', label="MNIST",
               alpha=0.1)
    
    ax.set_xlabel('Mean')
    ax.set_ylabel('Var')


def populate_layer_stats(model):
    X, y = load_mnist()

    img = torch.Tensor(X[:32]).unsqueeze(0).cuda()

    model.clear_output()
    model.register_hooks()
    model.evaluation_mode=True

    with autocast():
        out = model(img)

    model.evaluation_mode=False
    model.remove_hooks()



# def adjust_image_mean(image, desired_mean):
#     # Calculate current mean
#     current_mean = np.mean(image)

#     # Compute the difference between desired and current means
#     mean_diff = desired_mean - current_mean

#     # Subtract the difference from each pixel value to adjust the mean
#     adjusted_image = image - mean_diff

#     # Normalize pixel values between 0 and 1
#     # adjusted_image = adjusted_image.astype(np.float32)

#     return adjusted_image


# def disentangled_mean_plot(model, var_pert=False, inhibitory=False, return_acc=False, max_noise=1):

#     batch_size=32
#     if var_pert:
#         components = np.linspace(0.1, 5, 20)
#     else:
#         components = np.linspace(0, max_noise, 20)

#     vars = np.zeros(len(components))
#     mus = np.zeros(len(components))
#     if return_acc: acc = []

#     dta = get_mnist_dataloaders()['train']
#     images = []
#     for batch_idx, (data, _) in enumerate(dta):
#         images.extend(data)
#         if len(images) >= 32:
#             break
#     images = torch.stack(images)
    
#     for idx, mu in enumerate(components):
#         # Additive Gaussian noise with mean=0 and std=0.1
#         if var_pert:
#             noise = np.random.normal(loc=0, scale=mu, size=(784))
#         else:
#             noise = np.random.normal(loc=mu, scale=0, size=(784))

    
#         for img_idx in range(batch_size):

#             img = images[img_idx] + torch.Tensor(noise).cuda()

#             model.clear_output()
#             model.register_hooks()

#             with autocast():
#                 model(img)

#             layer_output = np.array(getattr(model, f"fc1_output"))
#             if inhibitory:
#                 vars[idx]+= layer_output[0,5]
#                 mus[idx]+= layer_output[0,4]
#             else:
#                 vars[idx]+= layer_output[0,2]
#                 mus[idx]+= layer_output[0,0]

#             model.remove_hooks()
#             model.clear_output()

#         if return_acc: acc.append(eval_model(model, noise=noise))
        
#     vars=vars/batch_size
#     mus=mus/batch_size

#     if return_acc: return components, mus, vars, acc

#     return components, mus, vars


# def get_mnist_dataloaders(transforms=None, n_mnist_distractors=0, batch_size=32):
#     mnist_path = "/network/datasets/mnist.var/mnist_torchvision/MNIST/processed"
#     eval_batch_size = 10000
#     train_dataset = SparseMnistDataset(f"{mnist_path}/training.pt", n_mnist_distractors, transforms=transforms)
#     train_dataset_eval = SparseMnistDataset(f"{mnist_path}/training.pt", n_mnist_distractors)
#     test_dataset  = SparseMnistDataset(f"{mnist_path}/test.pt", n_mnist_distractors)
    
#     train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
#     train_dataloader_eval = DataLoader(train_dataset_eval, eval_batch_size, shuffle=True)
#     test_dataloader  = DataLoader(test_dataset, eval_batch_size, shuffle=False)

#     return {"train":train_dataloader, "train_eval":train_dataloader_eval, "test":test_dataloader}

# def eval_model(model, loaders=get_mnist_dataloaders(), noise=None):
#     train_correct, n_train, train_loss, train_local_loss = 0., 0., 0., 0.
#     for idx, (ims, labs) in enumerate(loaders['train']):
#         with autocast():
#             if noise is not None:
#                 image_transform = ims + torch.Tensor(noise).cuda()
#             else:
#                 image_transform = ims
#             out = model(image_transform)
#             train_correct += out.argmax(1).eq(labs).sum().cpu().item()
#             n_train += ims.shape[0]
#         if idx == 300:
#             break
#     train_acc = train_correct / n_train * 100

#     return train_acc
