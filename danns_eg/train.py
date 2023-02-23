"""
https://mila-umontreal.slack.com/archives/CFAS8455H/p1659464809056259?thread_ts=1659386269.124849&cid=CFAS8455H
https://gist.github.com/lebrice/4a67df47d9fca3e199d3e7686396240c
FFCV version (Slower with PL, faster with pytorch): https://gist.github.com/lebrice/37d89c29388d7fc9ce267eed1ba6dbda

Issues:
    - seems to be the same dataloader size for testset and not with test set... imagenet
    - at leas the train set does not seem to get any larger when we don't reserve
    - lets look tomorrow! and start with comparing the standard nimagenet first. 

Look at this too - for the handling of test set? 
https://github.com/libffcv/ffcv-imagenet/blob/main/train_imagenet.py
https://github.com/Lightning-AI/lightning-bolts/blob/86f5d0937f9c3f23ca37de6ddd7c820f0e113193/pl_bolts/datamodules/imagenet_datamodule.py#L211
Tomorrow train cifar10 - repeat the arna example? 


https://myrtle.ai/learn/how-to-train-your-resnet/

This series seems cool.
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


import torch 
import torch.nn as nn 
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler, Adam

from data.dataloaders import get_dataloaders
#from data.imagenet_ffcv import ImagenetFfcvDataModule, IMAGENET_MEAN
import utils as train_utils
#from config import DANNS_DIR
#import lib.utils
from sequential import Sequential
#import models.kakaobrain as kakaobrain
import resnets
from optimisation import AdamW, get_linear_schedule_with_warmup, SGD
import optimisation as optimizer_utils

import torch.backends.xnnpack
print("XNNPACK is enabled: ", torch.backends.xnnpack.enabled, "\n")
DANNS_DIR = "/home/mila/r/roy.eyono/danns_eg/danns_eg" # BAD CODE: This needs to be changed

# Concatenate the path to the current file with the path to the danns_eg directory
SCALE_DIR = f"{DANNS_DIR}/scale_exps"



Section('train', 'Training related parameters').params(
    dataset=Param(str, 'dataset', default='cifar10'),
    batch_size=Param(int, 'batch-size', default=512),
    epochs=Param(int, 'epochs', default=100), 
    seed=Param(int, 'seed', default=7),
    use_testset=Param(bool, 'use testset as val set', default=False),
    )

Section('data', 'dataset related parameters').params(
    subtract_mean=Param(bool, 'subtract mean from the data', default=False),
) 
# note this should be false for danns bec i input could be positive if not
# we require x to be non-negative

Section('model', 'Model Parameters').params(
    name=Param(str, 'model to train', default='resnet50'),
    normtype=Param(str,'norm layer type - can be None', default='bn'),
    is_dann=Param(bool,'network is a dan network', default=True),
    n_outputs=Param(int,'e.g number of target classes', default=10),
    #input_shape=Param(tuple,'optional, none batch' 
)
Section('opt', 'optimiser parameters').params(
    algorithm=Param(str, 'learning algorithm', default='SGD'),
    exponentiated=Param(bool,'eg vs gd', default=False),
    wd=Param(float,'weight decay lambda', default=5e-4),
    momentum=Param(float,'momentum factor', default=0.0),
    lr=Param(float, 'lr and Wex if dann', default=0.0005),
    use_sep_inhib_lrs=Param(bool,' ', default=True),
    use_sep_bias_gain_lrs=Param(bool,' ', default=False),
    eg_normalise=Param(bool,'maintain sum of weights exponentiated is true ', default=False),
    nesterov=Param(bool, 'bool for nesterov momentum', False)
)

Section('opt.inhib_lrs').enable_if(lambda cfg:cfg['opt.use_sep_inhib_lrs']==True).params(
    wei=Param(float,'lr for Wei if dann', default=0.005),
    wix=Param(float,'lr for Wix if dann', default=0.005),
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
    wandb_tag=Param(str, 'tag under which to log runs', default="default"),
    save_results=Param(bool,'save_results', default=False)
)

def get_optimizer(p, model):
    params_groups = optimizer_utils.get_param_groups(model, return_groups_dict=True)
    # first construct the iterable of param groups
    parameters = []
    for k, group in params_groups.items():
        d = {"params":group, "name":k}
        
        if k in ["wix_params", "wei_params",'wex_params']: 
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
                   momentum=p.opt.momentum) #,exponentiated_grad=p.opt.exponentiated) 
        opt.nesterov = p.opt.nesterov
        # opt.eg_normalise = p.opt.eg_normalise
        return opt

    elif p.opt.algorithm.lower() == "adamw":
        #  this should be adapted in future for adamw specific args! 
        return AdamW(parameters, lr = p.opt.lr,
                     weight_decay=p.opt.wd,
                     exponentiated_grad=p.opt.exponentiated) 

def train_epoch(ep_i):
    # the vars below should all be defined in the global scope
    epoch_correct, total_ims = 0, 0
    for ims, labs in loaders['train']:
        model.train()
        opt.zero_grad(set_to_none=True)
        with autocast():
            out = model(ims)
            loss = loss_fn(out, labs)
            batch_correct = out.argmax(1).eq(labs).sum().cpu().item()
            batch_acc = batch_correct / ims.shape[0] * 100
            epoch_correct += batch_correct
            total_ims += ims.shape[0]

        if p.exp.use_wandb: 
            wandb.log({"update_acc":batch_acc,
                       "update_loss":loss.cpu().item(),
                       "lr":opt.param_groups[0]['lr']})

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        try: scheduler.step()
        except NameError: pass

    epoch_acc = epoch_correct / total_ims * 100
    results["online_epoch_acc"] = epoch_acc

def eval_model(ep_i):
    model.eval()
    with torch.no_grad():
        train_correct, n_train, train_loss = 0., 0., 0.
        for ims, labs in loaders['train_eval']:
            with autocast():
                out = model(ims)
                train_loss += loss_fn_sum(out, labs)
                train_correct += out.argmax(1).eq(labs).sum().cpu().item()
                n_train += ims.shape[0]
        train_acc = train_correct / n_train * 100
        train_loss /=  n_train
        
        test_correct, n_test, test_loss = 0., 0., 0.
        for ims, labs in loaders['test']:
            with autocast():
                # out = (model(ims) + model(ch.fliplr(ims))) / 2. # Test-time augmentation
                out = model(ims)
                test_loss += loss_fn_sum(out, labs)
                test_correct += out.argmax(1).eq(labs).sum().cpu().item()
                n_test += ims.shape[0]
        #print("Not currently running train eval!")
        test_acc = test_correct / n_test * 100
        test_loss /=  n_test

        results["test_accs"].append(test_acc)
        results["test_losses"].append(test_loss.cpu().item())
        results["train_accs"].append(train_acc)
        results["train_losses"].append(train_loss.cpu().item())
        results["ep_i"].append(ep_i)

        if p.exp.use_wandb: 
            wandb.log({"epoch_i":ep_i,
                "test_loss":results["test_losses"][-1], "test_acc":results["test_accs"][-1],
                "train_loss":results["train_losses"][-1], "train_acc":results["train_accs"][-1]})


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
    #model = cifar_resnet50(p)
    #model = kakaobrain.construct_KakaoBrain_model()
    model = resnets.resnet9_kakaobrain(p)
    #model = resnet.resnet18(p)
    return model

#%%
if __name__ == "__main__":
    print(f"We're at: {os.getcwd()}")
    # %%
    # python train.py --config-file=exp_configs/resnet18.yaml
    # python -m pdb -c continue train.py --config-file=exp_configs/resnet18.yaml
    # mount scratch: sshfs cornforj@login.server.mila.quebec:/network/scratch/c/cornforj ~/mnt/mila -p 2222
    device = train_utils.get_device()
    param_search_metrics = [] # for use in hyperparam searches, e.g list of sum of test loss? for seeds 
    for seed in [7, 8, 9]:# 8, 10]:
        results = {"test_accs" :[], "test_losses": [], 
                   "train_accs" :[], "train_losses":[], "ep_i":[],
                   "model_norm":[]}

        p = train_utils.get_config()
        p.train.seed = seed
        train_utils.set_seed_all(p.train.seed)

        if p.exp.use_wandb:
            os.environ['WANDB_DIR'] = str(Path.home()/ "scratch/")
            if p.exp.wandb_project == "": p.exp.wandb_project  = "220823_scale"
            params_to_log = train_utils.get_params_to_log_wandb(p)
            run = wandb.init(reinit=False, project=p.exp.wandb_project,
                             config=params_to_log, tags=[p.exp.wandb_tag])
            name = f'{"EG" if p.opt.exponentiated else "SGD"} lr:{p.opt.lr} wd:{p.opt.wd} m:{p.opt.momentum} '
            if p.model.is_dann:
                name += f"wix:{p.opt.inhib_lrs.wix} wei:{p.opt.inhib_lrs.wei}" 
            run.name = name
            print("run name", name)

        loaders = get_dataloaders(p)
        # %%
        model = build_model(p).cuda()
        params_groups = optimizer_utils.get_param_groups(model, return_groups_dict=True)


        #%%
        #print(model)
        EPOCHS = p.train.epochs
        opt = get_optimizer(p, model)

        # lr = 0.5
        # weight_decay = 5e-4
        # opt = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        
        iters_per_epoch = len(loaders['train']) 
        #iters_per_epoch = 50000 // 512
        print(iters_per_epoch)
        # so reaches peak at 5th epoch and then decays back down
        lr_schedule = np.interp(np.arange((EPOCHS+1) * iters_per_epoch),
                                [0, 5 * iters_per_epoch, EPOCHS * iters_per_epoch],
                                [0, 1, 0])
        plt.plot(lr_schedule)
        plt.show()
        
        scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
        scaler = GradScaler() 
        # https://wandb.ai/wandb_fc/tips/reports/How-to-Use-GradScaler-in-PyTorch--VmlldzoyMTY5MDA5
        # https://youtu.be/IkeEadgSy6w
        loss_fn = CrossEntropyLoss(label_smoothing=0.1)
        loss_fn_sum = CrossEntropyLoss(label_smoothing=0.1, reduction='sum')
        # %%
        results = train_model(p)
        param_search_metrics.append(np.sum(results["test_losses"]))

        if p.exp.use_wandb:
            run.summary["test_loss_auc"] = np.sum(results["test_losses"])
            run.finish()

        if p.exp.save_results:
            pass


        # you need to save the the results! 

    print(f"Reporting to orion mean sum of test losses {np.mean(param_search_metrics)}")
    report_objective(np.mean(param_search_metrics))



    


    





