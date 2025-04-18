"""
Here we train the DANNs network on the split MNIST task.

The goal is to evaluate the performance of layer normalization for DANNs networks.
"""
# %%
# %load_ext autoreload
# %autoreload 2
from operator import truediv
#import cv2
import os
import sys
import numpy as np 
from pprint import pprint
from tqdm import tqdm
import argparse
from fastargs import Section, Param, get_current_config

import matplotlib.pyplot as plt
import wandb
# from orion.client import report_objective
from pathlib import Path
import json
from fastargs.dict_utils import NestedNamespace
import pickle as pkl
import uuid


import torch 
import torch.nn as nn 
from torch.amp import GradScaler, autocast
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
import danns_eg.predictivernn as predictivernn
import danns_eg.drnn as drnn
from danns_eg.optimisation import AdamW, get_linear_schedule_with_warmup, SGD
import danns_eg.optimisation as optimizer_utils
#from munch import DefaultMunch
import danns_eg.utils as utils
#ood_scatter_plot, recon_var_plot, mean_plot, var_plot

import torch.backends.xnnpack
print("XNNPACK is enabled: ", torch.backends.xnnpack.enabled, "\n")
DANNS_DIR = "/home/mila/r/roy.eyono/danns_eg" # BAD CODE: This needs to be changed

# Concatenate the path to the current file with the path to the danns_eg directory
SCALE_DIR = f"{DANNS_DIR}/scale_exps"



Section('train', 'Training related parameters').params(
    dataset=Param(str, 'dataset', default='fashionmnist'),
    batch_size=Param(int, 'batch-size', default=128), #32
    epochs=Param(int, 'epochs', default=50), 
    seed=Param(int, 'seed', default=0),
    use_testset=Param(bool, 'use testset as val set', default=True),
    )

Section('data', 'dataset related parameters').params(
    subtract_mean=Param(bool, 'subtract mean from the data', default=False),
    brightness_factor=Param(float, 'random brightness jitter', default=0.75),
    brightness_factor_eval=Param(float, 'brightness evaluation', default=0),
    contrast_jitter=Param(bool, 'contrast jitter', default=False),

) 
# note this should be false for danns bec i input could be positive if not
# we require x to be non-negative

Section('model', 'Model Parameters').params(
    name=Param(str, 'model to train', default='resnet50'),
    normtype=Param(int,'train model with layernorm', default=0),
    is_dann=Param(int,'network is a dan network', default=1),  # This is a flag to indicate if the network is a dann network
    n_outputs=Param(int,'e.g number of target classes', default=10),
    homeostasis=Param(int,'homeostasis', default=1),
    excitation_training=Param(int,'training excitatory layers', default=1),
    implicit_homeostatic_loss=Param(int,'homeostasic loss', default=0),
    task_opt_inhib=Param(int,'train inhibition model on task loss', default=0),
    homeo_opt_exc=Param(int,'train excitatatory weights on inhibitory loss', default=0),
    homeostatic_annealing=Param(int,'applying annealing to homeostatic loss', default=0),
    hidden_layer_width=Param(int,'number of hidden layers', default=500),
    #input_shape=Param(tuple,'optional, none batch' 
)
Section('opt', 'optimiser parameters').params(
    algorithm=Param(str, 'learning algorithm', default='sgd'),
    exponentiated=Param(bool,'eg vs gd', default=False),
    wd=Param(float,'weight decay lambda', default=0), #0.001 # Weight decay is very bad for inhibition
    momentum=Param(float,'momentum factor', default=0), #0.5 # We need a seperate momentum for the inhib component as well
    inhib_momentum=Param(float,'inhib momentum factor', default=0),
    lr=Param(float, 'lr and Wex if dann', default=0.01), # Ideal learning rate for Vanilla RNN is 0.01
    use_sep_inhib_lrs=Param(int,' ', default=1),
    use_sep_bias_gain_lrs=Param(int,'add gain and bias to layer', default=0),
    eg_normalise=Param(bool,'maintain sum of weights exponentiated is true ', default=False),
    nesterov=Param(bool, 'bool for nesterov momentum', False),
    lambda_homeo=Param(float, 'lambda homeostasis', default=1),
    lambda_homeo_var=Param(float, 'lambda homeostasis', default=1),
)

Section('opt.inhib_lrs').enable_if(lambda cfg:cfg['opt.use_sep_inhib_lrs']==1).params(
    wei=Param(float,'lr for Wei if dann', default=0.001), # 0.001
    wix=Param(float,'lr for Wix if dann', default=0.001), # 0.1
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
    use_wandb=Param(int, 'flag to use wandb', default=0),
    wandb_project=Param(str, 'project under which to log runs', default="Test"),
    wandb_entity=Param(str, 'team under which to log runs', default=""),
    wandb_tag=Param(str, 'tag under which to log runs', default="default"),
    save_results=Param(bool,'save_results', default=False),
    save_model=Param(int, 'save model', default=0),
    name=Param(str, 'name of the run', default="dann_project"),
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



def train_epoch(model, loaders, loss_fn, local_loss_fn, opt, p, scaler, epoch):
    # the vars below should all be defined in the global scope
    epoch_correct, total_ims = 0, 0
    n_rows = 28 # MNIST / FashionMNIST sequential input
    for ims, labs in loaders['train']:


        model.train()
        opt.zero_grad(set_to_none=True)
        with autocast("cuda"):
            ims = ims.cuda(non_blocking=True)
            ims = ims.view(ims.shape[0], 1, 28, 28)
            labs = labs.cuda(non_blocking=True)
            model.reset_hidden(ims.shape[0])

            # Sequential input
            for row_i in range(n_rows):
                out, hidden_act  = model(ims[:, 0, row_i, :])
            
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
                    if 'Wix' in name or 'Wei' in name or 'Uix' in name or 'Uei' in name:
                        if 'fc_output' not in name:
                            if p.model.task_opt_inhib:
                                param.grad = torch.autograd.grad(scaler.scale(loss), param, retain_graph=True)[0] + param.grad
                            continue
                    
                    if p.model.excitation_training:
                        param.grad = torch.autograd.grad(scaler.scale(loss), param, retain_graph=True)[0]
        else:
            scaler.scale(loss).backward()

        if p.model.excitation_training:     
            scaler.step(opt)
            scaler.update()

    epoch_acc = epoch_correct / total_ims * 100
    results["online_epoch_acc"] = epoch_acc

def eval_model(epoch, model, loaders, loss_fn_sum, local_loss_fn, p):
    model.eval()
    n_rows = 28
    with torch.no_grad():
        train_correct, n_train, train_loss, train_local_loss = 0., 0., 0., 0.
        for ims, labs in loaders['train']:
            with autocast("cuda"):
                ims = ims.cuda()
                ims = ims.view(ims.shape[0], 1, 28, 28)
                labs = labs.cuda()
                num_of_local_layers = 0
                model.reset_hidden(ims.shape[0])
                # Sequential input
                for row_i in range(n_rows):
                    out, hidden_act  = model(ims[:, 0, row_i, :])
                loss_val = loss_fn_sum(out, labs)
                local_loss = local_loss_fn(hidden_act, p.opt.lambda_homeo, p.opt.lambda_homeo_var).item()
                train_loss += loss_val
                #print(f"Global Loss: {loss_val.item()}")
                train_correct += out.argmax(1).eq(labs).sum().cpu().item()
                n_train += ims.shape[0]
                train_local_loss += local_loss
        train_acc = train_correct / n_train * 100
        train_loss /=  n_train
        train_local_loss /= n_train
        
        model.register_eval=True
        test_correct, n_test, test_loss, test_local_loss = 0., 0., 0., 0.
        for ims, labs in loaders['test']:
            with autocast("cuda"):
                ims = ims.cuda()
                ims = ims.view(ims.shape[0], 1, 28, 28)
                labs = labs.cuda()
                # out = (model(ims) + model(ch.fliplr(ims))) / 2. # Test-time augmentation
                num_of_local_layers = 0
                model.reset_hidden(ims.shape[0])
                # Sequential input
                for row_i in range(n_rows):
                    out, hidden_act  = model(ims[:, 0, row_i, :])
                loss_val = loss_fn_sum(out, labs)
                local_loss = local_loss_fn(hidden_act, p.opt.lambda_homeo, p.opt.lambda_homeo_var).item()
                test_loss += loss_val
                #print(f"Global Loss: {loss_val.item()}")
                test_correct += out.argmax(1).eq(labs).sum().cpu().item()
                n_test += ims.shape[0]
                test_local_loss += local_loss

        model.register_eval=False
        #print("Not currently running train eval!")
        test_acc = test_correct / n_test * 100
        test_loss /=  n_test
        test_local_loss /= n_test

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

def build_model(p, scaler):
    #model = densenets.net(p)
    model = predictivernn.net(p, scaler)
    model.reset_hidden(batch_size=p.train.batch_size)
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
    local_loss_fn = drnn.LocalLossMean(p.model.hidden_layer_width, nonlinearity_loss=p.model.implicit_homeostatic_loss)

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
    scaler = GradScaler()
    model = build_model(p, scaler)
    opt = get_optimizer(p, model)
    model.set_optimizer(opt)
    model.register_hooks() # Register the forward hooks
    model = model.cuda()
    params_groups = optimizer_utils.get_param_groups(model, return_groups_dict=True)
    EPOCHS = p.train.epochs
    loss_fn = CrossEntropyLoss(label_smoothing=0.1)
    loss_fn_sum = CrossEntropyLoss(label_smoothing=0.1, reduction='sum')
    loss_fn_no_reduction = CrossEntropyLoss(label_smoothing=0.1, reduction='none')

    log_epochs = 10
    progress_bar = tqdm(range(1,1+(EPOCHS)))
    
    for epoch in progress_bar:
        train_epoch(model, loaders, loss_fn, local_loss_fn, opt, p, scaler, epoch)
        if epoch%1==0:
            eval_model(epoch, model, loaders, loss_fn_sum, local_loss_fn, p)          
            progress_bar.set_description("Train/test accuracy after {} epochs: {:.2f}/{:.2f}".format(epoch,results["train_accs"][-1],results["test_accs"][-1]))

    if p.exp.use_wandb:
        run.summary["test_loss_auc"] = np.sum(results["test_losses"])
        run.finish()

    model.remove_hooks()

    if p.exp.save_model:
        save_dir = f'/network/scratch/r/roy.eyono/{p.exp.name}_{p.train.dataset}_{p.data.brightness_factor}'
        os.makedirs(save_dir, exist_ok=True)
        best_axc=max(results["test_accs"])
        model_name = str(uuid.uuid4()) + f'_{best_axc}.pth'
        if p.exp.use_wandb:
            model_name = f'{run.name}.pth'
        model_path = os.path.join(save_dir, model_name)
        torch.save(model.state_dict(), model_path)

    # Print best test and training accuracy
    print("Best train accuracy: {:.2f}".format(max(results["train_accs"])))
    print("Best test accuracy: {:.2f}".format(max(results["test_accs"])))


