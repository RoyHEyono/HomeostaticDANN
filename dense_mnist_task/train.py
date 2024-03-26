"""
Here we train the DANNs network on the split MNIST task.

The goal is to evaluate the performance of layer normalization for DANNs networks.
"""
# %%
# %load_ext autoreload
# %autoreload 2
from operator import truediv
import cv2
import os
import sys
import numpy as np 
from pprint import pprint
from tqdm import tqdm
import argparse
from fastargs import Section, Param, get_current_config

import matplotlib.pyplot as plt
import wandb
from orion.client import report_objective
from pathlib import Path
import json
from fastargs.dict_utils import NestedNamespace
import pickle as pkl


import torch 
import torch.nn as nn 
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler, Adam

from danns_eg.data.dataloaders import get_dataloaders
#from data.imagenet_ffcv import ImagenetFfcvDataModule, IMAGENET_MEAN
import danns_eg.utils as train_utils
#from config import DANNS_DIR
#import lib.utils
from danns_eg.sequential import Sequential
#import models.kakaobrain as kakaobrain
import danns_eg.resnets as resnets
import danns_eg.densenet as densenets
from danns_eg.optimisation import AdamW, get_linear_schedule_with_warmup, SGD
import danns_eg.optimisation as optimizer_utils
from munch import DefaultMunch
from danns_eg.data.mnist import get_sparse_fashionmnist_dataloaders, get_sparse_mnist_dataloaders
from utils import ood_scatter_plot, recon_var_plot, mean_plot, var_plot

import torch.backends.xnnpack
print("XNNPACK is enabled: ", torch.backends.xnnpack.enabled, "\n")
DANNS_DIR = "/home/mila/r/roy.eyono/danns_eg" # BAD CODE: This needs to be changed

# Concatenate the path to the current file with the path to the danns_eg directory
SCALE_DIR = f"{DANNS_DIR}/scale_exps"



Section('train', 'Training related parameters').params(
    dataset=Param(str, 'dataset', default='mnist'),
    batch_size=Param(int, 'batch-size', default=32),
    epochs=Param(int, 'epochs', default=50), 
    seed=Param(int, 'seed', default=0),
    use_testset=Param(bool, 'use testset as val set', default=False),
    )

Section('data', 'dataset related parameters').params(
    subtract_mean=Param(bool, 'subtract mean from the data', default=False),
) 
# note this should be false for danns bec i input could be positive if not
# we require x to be non-negative

Section('model', 'Model Parameters').params(
    name=Param(str, 'model to train', default='resnet50'),
    normtype=Param(str,'norm layer type - can be None', default='ln'),
    is_dann=Param(bool,'network is a dan network', default=True),  # This is a flag to indicate if the network is a dann network
    n_outputs=Param(int,'e.g number of target classes', default=10),
    homeostasis=Param(bool,'homeostasis', default=True),
    #input_shape=Param(tuple,'optional, none batch' 
)
Section('opt', 'optimiser parameters').params(
    algorithm=Param(str, 'learning algorithm', default='SGD'),
    exponentiated=Param(bool,'eg vs gd', default=False),
    wd=Param(float,'weight decay lambda', default=0.001), #0.001 # Weight decay is very bad for inhibition
    momentum=Param(float,'momentum factor', default=0.5), #0.5 # We need a seperate momentum for the inhib component as well
    inhib_momentum=Param(float,'inhib momentum factor', default=0),
    lr=Param(float, 'lr and Wex if dann', default=0.01),
    use_sep_inhib_lrs=Param(bool,' ', default=True),
    use_sep_bias_gain_lrs=Param(bool,' ', default=False),
    eg_normalise=Param(bool,'maintain sum of weights exponentiated is true ', default=False),
    nesterov=Param(bool, 'bool for nesterov momentum', False),
    lambda_homeo=Param(float, 'lambda homeostasis', default=1),
    lambda_var=Param(float, 'lambda homeostasis', default=1),
)

Section('opt.inhib_lrs').enable_if(lambda cfg:cfg['opt.use_sep_inhib_lrs']==True).params(
    wei=Param(float,'lr for Wei if dann', default=0.5), # 0.001
    wix=Param(float,'lr for Wix if dann', default=0.5), # 0.1
)

Section('opt.bias_gain_lrs').enable_if(lambda cfg:cfg['opt.use_sep_bias_gain_lrs']==True).params(
    b=Param(float,'lr for bias', default=0.005),
    g=Param(float,'lr for gains', default=0.005),
) 

Section('exp', 'General experiment details').params(
    ckpt_dir=Param(str, 'ckpt-dir', default=f"{SCALE_DIR}/checkpoints"),
    num_workers=Param(int, 'num of CPU workers', default=4),
    use_autocast=Param(bool, 'autocast fp16', default=True),
    log_interval=Param(int, 'log-interval in terms of epochs', default=1),
    data_dir=Param(str, 'data dir - if passed will not write ffcv', default=''),
    use_wandb=Param(bool, 'flag to use wandb', default=False),
    wandb_project=Param(str, 'project under which to log runs', default=""),
    wandb_entity=Param(str, 'team under which to log runs', default=""),
    wandb_tag=Param(str, 'tag under which to log runs', default="default"),
    save_results=Param(bool,'save_results', default=False),
    save_model=Param(bool, 'save model', default=False),
    name=Param(str, 'name of the run', default=""),
) #TODO: Add wandb group param


def get_optimizer(p, model):
    params_groups = optimizer_utils.get_param_groups(model, return_groups_dict=True)
    # first construct the iterable of param groups
    parameters = []
    for k, group in params_groups.items():
        d = {"params":group, "name":k}
        
        if k in ['wex_params', 'wix_params', 'wei_params']: # I've removed the clamping from these: "wix_params", "wei_params"
            d["positive_only"] = True
        else: 
            d["positive_only"] = False
        
        if p.opt.use_sep_inhib_lrs:
            if k == "wix_params": d['lr'] = p.opt.inhib_lrs.wix
            elif k == "wei_params": d['lr'] = p.opt.inhib_lrs.wei
        
        if k == "norm_biases":
            d['exponentiated_grad'] = False 
        if p.opt.use_sep_bias_gain_lrs:
            if k == "norm_biases": 
                d['lr'] = p.opt.bias_gain_lrs.b
                print("hard coding non exp grad for biases")
            elif k == "norm_gains": d['lr'] = p.opt.bias_gain_lrs.g
        
        parameters.append(d)
    
    if p.opt.algorithm.lower() == "sgd":
        opt = SGD(parameters, lr = p.opt.lr,
                   weight_decay=p.opt.wd,
                   momentum=p.opt.momentum, inhib_momentum=p.opt.inhib_momentum) #,exponentiated_grad=p.opt.exponentiated)  
        opt.nesterov = p.opt.nesterov
        # opt.eg_normalise = p.opt.eg_normalise
        return opt

    elif p.opt.algorithm.lower() == "adamw":
        #  this should be adapted in future for adamw specific args! 
        return AdamW(parameters, lr = p.opt.lr,
                     weight_decay=p.opt.wd,
                     exponentiated_grad=p.opt.exponentiated) 



def train_epoch(model, loaders, loss_fn, opt, scheduler, p, scaler, epoch):
    # the vars below should all be defined in the global scope
    epoch_correct, total_ims = 0, 0
    for ims, labs in loaders['train']:


        model.train()
        opt.zero_grad(set_to_none=True)
        with autocast():
            out = model(ims)
            loss = loss_fn(out, labs)
            
            # TODO: We need to return the local loss from the model
            # local_loss = model(ims).ln_mu_loss

            batch_correct = out.argmax(1).eq(labs).sum().cpu().item()
            batch_acc = batch_correct / ims.shape[0] * 100
            epoch_correct += batch_correct
            total_ims += ims.shape[0]

        if p.exp.use_wandb: 
            wandb.log({"update_acc":batch_acc,
                       "update_loss":loss.cpu().item(),
                       "lr":opt.param_groups[0]['lr']})

        if p.model.homeostasis:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if 'Wix' in name or 'Wei' in name:
                        if 'fc5' not in name: # TEMP FIX: I'm training last layer inhib on task loss
                            continue
                    
                    param.grad = torch.autograd.grad(scaler.scale(loss), param, retain_graph=True)[0]

                    # if 'Wex' in name:
                    #     param.grad = None # This locks in the homeostatic loss only
                    
                    
        else:
            scaler.scale(loss).backward()
                    
        scaler.step(opt)
        scaler.update()
        try: scheduler.step()
        except NameError: pass

    epoch_acc = epoch_correct / total_ims * 100
    results["online_epoch_acc"] = epoch_acc

def eval_model(epoch, model, loaders, loss_fn_sum, p):
    model.eval()
    with torch.no_grad():
        train_correct, n_train, train_loss, train_local_loss = 0., 0., 0., 0.
        for ims, labs in loaders['train_eval']:
            with autocast():
                num_of_local_layers = 0
                out = model(ims)
                loss_val = loss_fn_sum(out, labs)
                train_loss += loss_val
                #print(f"Global Loss: {loss_val.item()}")
                train_correct += out.argmax(1).eq(labs).sum().cpu().item()
                n_train += ims.shape[0]
                train_local_loss += model.get_local_loss()
        train_acc = train_correct / n_train * 100
        train_loss /=  n_train
        train_local_loss /= n_train
        train_local_loss *= 1000 # To make it comparable to the global loss
        
        test_correct, n_test, test_loss, test_local_loss = 0., 0., 0., 0.
        for ims, labs in loaders['test']:
            with autocast():
                # out = (model(ims) + model(ch.fliplr(ims))) / 2. # Test-time augmentation
                num_of_local_layers = 0
                out = model(ims)
                loss_val = loss_fn_sum(out, labs)
                test_loss += loss_val
                #print(f"Global Loss: {loss_val.item()}")
                test_correct += out.argmax(1).eq(labs).sum().cpu().item()
                n_test += ims.shape[0]
                test_local_loss += model.get_local_loss()

        #print("Not currently running train eval!")
        test_acc = test_correct / n_test * 100
        test_loss /=  n_test
        test_local_loss /= n_test
        test_local_loss *= 1000 # To make it comparable to the global loss

        results["test_accs"].append(test_acc)
        results["test_losses"].append(test_loss.cpu().item())
        results["train_accs"].append(train_acc)
        results["train_losses"].append(train_loss.cpu().item())
        results["train_local_losses"].append(train_local_loss)
        results["test_local_losses"].append(test_local_loss)
        results["train_total_loss"].append(train_loss.cpu().item() + train_local_loss)
        results["test_total_loss"].append(test_loss.cpu().item() + test_local_loss)
        results["ep_i"].append(epoch)

        if p.exp.use_wandb: 
            wandb.log({"epoch_i":epoch,
                "test_loss":results["test_losses"][-1], "test_acc":results["test_accs"][-1],
                "train_loss":results["train_losses"][-1], "train_acc":results["train_accs"][-1],
                "train_local_loss":results["train_local_losses"][-1], "test_local_loss":results["test_local_losses"][-1],
                "train_total_loss":results["train_total_loss"][-1], "test_total_loss":results["test_total_loss"][-1],})


def train_model(p):
    log_epochs = 1 # how often to eval and log model performance
    #print("Training model for {} epochs with {} optimizer, lr={}, wd={:.1e}".format(EPOCHS,opt.__class__.__name__,lr,weight_decay))
    progress_bar = tqdm(range(1,1+p.train.epochs))
    
    for ep_i in progress_bar:
        train_epoch(ep_i)
        if ep_i%log_epochs==0:
            eval_model(ep_i)
            # here also get model info like % dead units, "effective rank", and weight norm           
            progress_bar.set_description("Train/test accuracy after {} epochs: {:.2f}/{:.2f}".format(
                ep_i,results["train_accs"][-1],results["test_accs"][-1]))
        #print(scheduler.get_last_lr())
    return results 

def build_model(p):
    model = densenets.net(p)
    return model

def convert_to_dict(obj):
    if isinstance(obj, NestedNamespace):
        obj_dict = obj.__dict__
        for key, value in obj_dict.items():
            if isinstance(value, NestedNamespace):
                obj_dict[key] = convert_to_dict(value)
        return obj_dict
    elif isinstance(obj, list):
        return [convert_to_dict(item) if isinstance(item, NestedNamespace) else item for item in obj]
    else:
        return obj

if __name__ == "__main__":
    print(f"We're at: {os.getcwd()}")
    device = train_utils.get_device()
    results = {"test_accs" :[], "test_losses": [], 
                "train_accs" :[], "train_losses":[], "ep_i":[],
                "model_norm":[], "train_local_losses":[], "test_local_losses":[],
                "train_total_loss":[], "test_total_loss":[]}

    p = train_utils.get_config()
    train_utils.set_seed_all(p.train.seed)

    if p.exp.use_wandb:
        os.environ['WANDB_DIR'] = str(Path.home()/ "scratch/")
        if p.exp.wandb_project == "": p.exp.wandb_project  = "220823_scale"
        params_to_log = train_utils.get_params_to_log_wandb(p)
        run = wandb.init(reinit=False, project=p.exp.wandb_project, entity=p.exp.wandb_entity,
                            config=params_to_log, tags=[p.exp.wandb_tag])
        name = f'{"EG" if p.opt.exponentiated else "SGD"} lr:{p.opt.lr} wd:{p.opt.wd} m:{p.opt.momentum} '
        if p.model.is_dann:
            name += f"wix:{p.opt.inhib_lrs.wix} wei:{p.opt.inhib_lrs.wei}"

    loaders = get_dataloaders(p)
    model = build_model(p)
    model.register_hooks() # Register the forward hooks
    model = model.cuda()
    params_groups = optimizer_utils.get_param_groups(model, return_groups_dict=True)
    EPOCHS = p.train.epochs
    opt = get_optimizer(p, model)
    iters_per_epoch = len(loaders['train'])
    lr_schedule = np.interp(np.arange((EPOCHS+1) * iters_per_epoch),
                            [0, 5 * iters_per_epoch, EPOCHS * iters_per_epoch],
                            [0, 1, 0])
    
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(label_smoothing=0.1)
    loss_fn_sum = CrossEntropyLoss(label_smoothing=0.1, reduction='sum')
    loss_fn_no_reduction = CrossEntropyLoss(label_smoothing=0.1, reduction='none')

    log_epochs = 10
    progress_bar = tqdm(range(1,1+(EPOCHS)))
    
    for epoch in progress_bar:
        train_epoch(model, loaders, loss_fn, opt, scheduler, p, scaler, epoch)
        if epoch%1==0:
            eval_model(epoch, model, loaders, loss_fn_sum, p)          
            progress_bar.set_description("Train/test accuracy after {} epochs: {:.2f}/{:.2f}".format(epoch,results["train_accs"][-1],results["test_accs"][-1]))

    if p.exp.use_wandb:
        run.summary["test_loss_auc"] = np.sum(results["test_losses"])
        run.finish()

    # Save pickle of layers
    layers_dict = []
    for layer_num in range(1,2):
        layers_dict.append(np.array(getattr(model, f"fc{layer_num}_output")))

    if not p.exp.name == "":

        # Open the file in binary write mode ('wb')
        with open(f"{p.exp.name}_layers.pkl", 'wb') as file:
            # Use pickle.dump() to save the list to the file
            pkl.dump(layers_dict, file)

        with open(f"{p.exp.name}_results.pkl", 'wb') as file:
            # Use pickle.dump() to save the list to the file
            pkl.dump(results, file)

        torch.save(model.state_dict(), f"{p.exp.name}.pt")


        # Generate plots here
        mean_plot(model)
        var_plot(model)
        
        #recon_var_plot(model)
        ood_scatter_plot(p, loaders, model)

    # Print best test and training accuracy
    print("Best train accuracy: {:.2f}".format(max(results["train_accs"])))
    print("Best test accuracy: {:.2f}".format(max(results["test_accs"])))

    

    



    


    






# %%
