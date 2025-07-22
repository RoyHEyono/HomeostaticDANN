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
from fastargs.decorators import param
import uuid

import matplotlib.pyplot as plt
import wandb
from orion.client import report_objective
from pathlib import Path
import json
from fastargs.dict_utils import NestedNamespace
import pickle as pkl


import torch 
import torch.nn as nn 
from torch.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler, Adam
import torch.nn.functional as F

from danns_eg.data.dataloaders import get_dataloaders
#from data.imagenet_ffcv import ImagenetFfcvDataModule, IMAGENET_MEAN
import danns_eg.utils as train_utils
#import models.kakaobrain as kakaobrain
import danns_eg.homeostaticdensenet as homeostaticdensenet
import danns_eg.homeostatic_mu_var_densenet as fullhomeostaticdensenet
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
    batch_size=Param(int, 'batch-size', default=32),
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

Section('model', 'Model Parameters').params(
    name=Param(str, 'model to train', default='resnet50'),
    normtype=Param(int,'train model with layernorm', default=0),
    normtype_detach=Param(int,'train model with detached layernorm', default=0),
    is_dann=Param(int,'network is a dan network', default=1),  # This is a flag to indicate if the network is a dann network
    n_outputs=Param(int,'e.g number of target classes', default=10),
    homeostasis=Param(int,'homeostasis', default=1),
    feedback_alignment=Param(int,'homeostasis with feedback alignment', default=0),
    ln_feedback=Param(str, 'feedback for ln gradient', default='full'),
    shunting=Param(int,'divisive inhibition', default=1),
    excitation_training=Param(int,'training excitatory layers', default=1),
    implicit_homeostatic_loss=Param(int,'homeostasic loss', default=0),
    task_opt_inhib=Param(int,'train inhibition model on task loss', default=0),
    homeo_opt_exc=Param(int,'train excitatatory weights on inhibitory loss', default=0),
    homeostatic_annealing=Param(int,'applying annealing to homeostatic loss', default=0),
    hidden_layer_width=Param(int,'number of hidden layers', default=234),
    #input_shape=Param(tuple,'optional, none batch' 
)
Section('opt', 'optimiser parameters').params(
    algorithm=Param(str, 'learning algorithm', default='sgd'),
    exponentiated=Param(bool,'eg vs gd', default=False),
    wd=Param(float,'weight decay lambda', default=0), #0.001 # Weight decay is very bad for inhibition
    momentum=Param(float,'momentum factor', default=0), #0.5 # We need a seperate momentum for the inhib component as well
    inhib_momentum=Param(float,'inhib momentum factor', default=0),
    lr=Param(float, 'lr and Wex if dann', default=0.0098),
    use_sep_inhib_lrs=Param(int,' ', default=1),
    use_sep_bias_gain_lrs=Param(int,'add gain and bias to layer', default=1),
    eg_normalise=Param(bool,'maintain sum of weights exponentiated is true ', default=False),
    nesterov=Param(bool, 'bool for nesterov momentum', False),
    lambda_homeo=Param(float, 'lambda homeostasis', default=0.001), #0.001
    lambda_homeo_var=Param(float, 'lambda homeostasis', default=100),
)

Section('opt.inhib_lrs').enable_if(lambda cfg:cfg['opt.use_sep_inhib_lrs']==1).params(
    wei=Param(float,'lr for Wei if dann', default=0.00294), # 0.001
    wix=Param(float,'lr for Wix if dann', default=0.64646), # 0.1
)

Section('opt.bias_gain_lrs').enable_if(lambda cfg:cfg['opt.use_sep_bias_gain_lrs']==True).params(
    b=Param(float,'lr for bias', default=0.5),
    g=Param(float,'lr for gains', default=0.5),
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
)

def train_epoch(model, loaders, loss_fn, opt, p, scaler):
    # the vars below should all be defined in the global scope
    epoch_correct, total_ims = 0, 0
    for _, (ims, labs) in enumerate(loaders['train']):


        model.train()
        opt.zero_grad(set_to_none=True)
          
        with autocast("cuda"):
            ims, labs = ims.squeeze(1).cuda(), labs.cuda()
            out = model(ims)
            loss = loss_fn(out, labs)

            batch_correct = out.argmax(1).eq(labs).sum().cpu().item()
            batch_acc = batch_correct / ims.shape[0] * 100
            epoch_correct += batch_correct
            total_ims += ims.shape[0]

        grad_norms = {}
        weight_norms = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if ('ix' in name or 'ei' in name) and 'fc_output' not in name:
                    continue

                if p.model.excitation_training:
                    param.grad = torch.autograd.grad(scaler.scale(loss), param, retain_graph=True)[0]
                    grad_norm = param.grad.norm(2).item()  # L2 norm
                    grad_norms[f"grad_norm/{name}"] = grad_norm
                    weight_norm = param.norm(2).item()  # L2 norm of the weights
                    weight_norms[f"weight_norm/{name}"] = weight_norm
                else:
                    param.grad = torch.autograd.grad(scaler.scale(loss*0), param, retain_graph=True)[0]
        
        if p.exp.use_wandb: 
            wandb.log({"update_acc":batch_acc,
                       "update_loss":loss.cpu().item(),
                       "lr":opt.param_groups[0]['lr']})
            wandb.log(grad_norms)
        scaler.step(opt)
        scaler.update()

    epoch_acc = epoch_correct / total_ims * 100
    results["online_epoch_acc"] = epoch_acc

def eval_model(epoch, model, loaders, loss_fn_sum, p):
    model.eval()
    with torch.no_grad():
        train_correct, n_train, train_loss, train_local_loss = 0., 0., 0., 0.
        for ims, labs in loaders['train']:
            with autocast("cuda"):
                ims, labs = ims.squeeze(1).cuda(), labs.cuda()
                out = model(ims)
                loss_val = loss_fn_sum(out, labs)
                train_loss += loss_val
                local_loss = model.get_local_val()
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
                ims, labs = ims.squeeze(1).cuda(), labs.cuda()
                out = model(ims)
                loss_val = loss_fn_sum(out, labs)
                test_loss += loss_val
                local_loss = model.get_local_val()
                test_correct += out.argmax(1).eq(labs).sum().cpu().item()
                n_test += ims.shape[0]
                test_local_loss += local_loss

        model.register_eval=False
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
                "train_loss":results["train_losses"][-1], "train_acc":results["train_accs"][-1],})

def build_model(p, scaler):
    if p.model.shunting:
        model = fullhomeostaticdensenet.net(p, scaler)
    else:
        model = homeostaticdensenet.net(p, scaler)
    return model

class DummyGradScaler:
    def scale(self, loss):
        return loss  # Returns the loss unchanged
    
    def step(self, optimizer):
        optimizer.step()  # Just calls optimizer.step() without scaling
    
    def update(self):
        pass  # No-op

    def unscale_(self, optimizer):
        pass  # No-op

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
            if p.opt.use_sep_inhib_lrs:
                name += f"wix:{p.opt.inhib_lrs.wix} wei:{p.opt.inhib_lrs.wei}"
            else:
                name += f"wix:{p.opt.lr} wei:{p.opt.lr}"

    loaders = get_dataloaders(p)
    scaler = DummyGradScaler() #GradScaler()
    model = build_model(p, scaler)
    model.register_hooks() # Register the forward hooks
    model = model.cuda()
    params_groups = optimizer_utils.get_param_groups(model, return_groups_dict=True)
    EPOCHS = p.train.epochs
    opt = optimizer_utils.get_optimizer(p, model)
    iters_per_epoch = len(loaders['train'])
    
    loss_fn = CrossEntropyLoss(label_smoothing=0.1)
    loss_fn_sum = CrossEntropyLoss(label_smoothing=0.1, reduction='sum')
    loss_fn_no_reduction = CrossEntropyLoss(label_smoothing=0.1, reduction='none')

    log_epochs = 10
    progress_bar = tqdm(range(1,1+(EPOCHS)))
    
    for epoch in progress_bar:
        train_epoch(model, loaders, loss_fn, opt, p, scaler)
        if epoch%1==0:
            eval_model(epoch, model, loaders, loss_fn_sum, p)          
            progress_bar.set_description(
            "Epoch {} | Train/Test Accuracy: {:.2f}/{:.2f} | Local Loss: {:.2e}/{:.2e}".format(
                epoch,
                results["train_accs"][-1],
                results["test_accs"][-1],
                results["train_local_losses"][-1],
                results["test_local_losses"][-1]
            )
            )

    if p.exp.use_wandb:
        run.summary["test_loss_auc"] = np.sum(results["test_losses"])
        run.finish()

    model.remove_hooks()

    # Print best test and training accuracy
    print("Best train accuracy: {:.2f}".format(max(results["train_accs"])))
    print("Best test accuracy: {:.2f}".format(max(results["test_accs"])))


