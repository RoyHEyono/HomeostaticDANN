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

from danns_eg.data.dataloaders import get_dataloaders
#from data.imagenet_ffcv import ImagenetFfcvDataModule, IMAGENET_MEAN
import danns_eg.utils as train_utils
#from config import DANNS_DIR
#import lib.utils
from danns_eg.sequential import Sequential
import danns_eg.dense as densenn
#import models.kakaobrain as kakaobrain
import danns_eg.resnets as resnets
import danns_eg.deepdensenet as deepdensenets
import danns_eg.predictivernn as predictivernn
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
# note this should be false for danns bec i input could be positive if not
# we require x to be non-negative

Section('model', 'Model Parameters').params(
    name=Param(str, 'model to train', default='resnet50'),
    normtype=Param(int,'train model with layernorm', default=1),
    normtype_detach=Param(int,'train model with detached layernorm', default=0),
    is_dann=Param(int,'network is a dan network', default=1),  # This is a flag to indicate if the network is a dann network
    n_outputs=Param(int,'e.g number of target classes', default=10),
    homeostasis=Param(int,'homeostasis', default=0),
    shunting=Param(int,'divisive inhibition', default=0),
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
    lr=Param(float, 'lr and Wex if dann', default=0.01),
    use_sep_inhib_lrs=Param(int,' ', default=1),
    use_sep_bias_gain_lrs=Param(int,'add gain and bias to layer', default=0),
    eg_normalise=Param(bool,'maintain sum of weights exponentiated is true ', default=False),
    nesterov=Param(bool, 'bool for nesterov momentum', False),
    lambda_homeo=Param(float, 'lambda homeostasis', default=0.01), #0.001
    lambda_homeo_var=Param(float, 'lambda homeostasis', default=1),
)

Section('opt.inhib_lrs').enable_if(lambda cfg:cfg['opt.use_sep_inhib_lrs']==1).params(
    wei=Param(float,'lr for Wei if dann', default=0.01), # 0.001
    wix=Param(float,'lr for Wix if dann', default=0.01), # 0.1
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
        
        if k in ['wex_params', 'wix_params', 'wei_params']:
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
            elif k == "norm_gains":
                d['lr'] = p.opt.bias_gain_lrs.g
                print("hard coding non exp grad for biases")
        
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
    annealing_temp = 0
    time_step_max =  len(loaders['train']) * 50
    idx_batch_count = len(loaders['train']) * epoch
    for idx_batch, (ims, labs) in enumerate(loaders['train']):


        model.train()
        opt.zero_grad(set_to_none=True)
          

        with autocast("cuda"):
            ims, labs = ims.squeeze(1).cuda(), labs.cuda()
            out, hidden_act = model(ims)
            loss = loss_fn(out, labs)
            local_loss, local_loss_val = local_loss_fn(hidden_act, p.opt.lambda_homeo, p.opt.lambda_homeo_var)

            batch_correct = out.argmax(1).eq(labs).sum().cpu().item()
            batch_acc = batch_correct / ims.shape[0] * 100
            epoch_correct += batch_correct
            total_ims += ims.shape[0]
        
        if p.model.homeostatic_annealing:
            annealing_temp = utils.cosine_annealing(idx_batch_count, time_step_max)
            model.set_homeostatic_temp(1-annealing_temp)
            loss = annealing_temp * loss

        
        
        grad_norms = {}
        weight_norms = {}

        if p.model.homeostasis:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if 'Wix' in name or 'Wei' in name or 'alpha' in name:
                        if 'fc_output' not in name:
                            if p.model.task_opt_inhib:
                                param.grad = torch.autograd.grad(scaler.scale(loss), param, retain_graph=True)[0] + torch.autograd.grad(scaler.scale(local_loss), param, retain_graph=True)[0]
                            else:
                                param.grad = torch.autograd.grad(scaler.scale(local_loss), param, retain_graph=True)[0]
                            grad_norm = param.grad.norm(2).item()  # L2 norm
                            grad_norms[f"grad_norm/{name}"] = grad_norm
                            weight_norm = param.norm(2).item()  # L2 norm of the weights
                            weight_norms[f"weight_norm/{name}"] = weight_norm
                            continue
                    
                    if p.model.excitation_training:
                        param.grad = torch.autograd.grad(scaler.scale(loss), param, retain_graph=True)[0]
                        grad_norm = param.grad.norm(2).item()  # L2 norm
                        grad_norms[f"grad_norm/{name}"] = grad_norm
                        weight_norm = param.norm(2).item()  # L2 norm of the weights
                        weight_norms[f"weight_norm/{name}"] = weight_norm
        else:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.grad = torch.autograd.grad(scaler.scale(loss), param, retain_graph=True)[0]
                    grad_norm = param.grad.norm(2).item()  # L2 norm
                    grad_norms[f"grad_norm/{name}"] = grad_norm
                    weight_norm = param.norm(2).item()  # L2 norm of the weights
                    weight_norms[f"weight_norm/{name}"] = weight_norm
            
            #scaler.scale(loss).backward()
        
        if p.exp.use_wandb: 
            wandb.log({"update_acc":batch_acc,
                       "update_loss":loss.cpu().item(),
                       "lr":opt.param_groups[0]['lr']})
            wandb.log(grad_norms)
        scaler.step(opt)
        scaler.update()
        idx_batch_count = idx_batch_count + 1

    epoch_acc = epoch_correct / total_ims * 100
    results["online_epoch_acc"] = epoch_acc

def eval_model(epoch, model, loaders, loss_fn_sum, local_loss_fn, p):
    model.eval()
    with torch.no_grad():
        train_correct, n_train, train_loss, train_local_loss = 0., 0., 0., 0.
        for ims, labs in loaders['train']:
            with autocast("cuda"):
                num_of_local_layers = 0
                ims, labs = ims.squeeze(1).cuda(), labs.cuda()
                out, hidden_act = model(ims)
                loss_val = loss_fn_sum(out, labs)
                train_loss += loss_val
                _, local_loss = local_loss_fn(hidden_act, p.opt.lambda_homeo, p.opt.lambda_homeo_var)
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
                # out = (model(ims) + model(ch.fliplr(ims))) / 2. # Test-time augmentation
                num_of_local_layers = 0
                ims, labs = ims.squeeze(1).cuda(), labs.cuda()
                out, hidden_act = model(ims)
                loss_val = loss_fn_sum(out, labs)
                test_loss += loss_val
                _, local_loss = local_loss_fn(hidden_act, p.opt.lambda_homeo, p.opt.lambda_homeo_var)
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


def train_model(p):
    log_epochs = 1 # how often to eval and log model performance
    #print("Training model for {} epochs with {} optimizer, lr={}, wd={:.1e}".format(EPOCHS,opt.__class__.__name__,lr,weight_decay))
    progress_bar = tqdm(range(1,1+p.train.epochs))
    
    for ep_i in progress_bar:
        train_epoch(ep_i)
        if ep_i%log_epochs==0:
            eval_model(ep_i)
            # here also get model info like % dead units, "effective rank", and weight norm           
            progress_bar.set_description(
            "Epoch {} | Train/Test Accuracy: {:.2f}/{:.2f} | Local Loss: {}/{}".format(
                ep_i,
                results["train_accs"][-1],
                results["test_accs"][-1],
                results["train_local_losses"][-1],
                results["test_local_losses"][-1]
            )
            )
        #print(scheduler.get_last_lr())
    return results 

def build_model(p):
    model = deepdensenets.net(p)
    #model = predictivernn.net(p)
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
            if p.opt.use_sep_inhib_lrs:
                name += f"wix:{p.opt.inhib_lrs.wix} wei:{p.opt.inhib_lrs.wei}"
            else:
                name += f"wix:{p.opt.lr} wei:{p.opt.lr}"

    loaders = get_dataloaders(p)
    model = build_model(p)
    model.register_hooks() # Register the forward hooks
    model = model.cuda()
    params_groups = optimizer_utils.get_param_groups(model, return_groups_dict=True)
    EPOCHS = p.train.epochs
    opt = get_optimizer(p, model)
    iters_per_epoch = len(loaders['train'])
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(label_smoothing=0.1)
    loss_fn_sum = CrossEntropyLoss(label_smoothing=0.1, reduction='sum')
    loss_fn_no_reduction = CrossEntropyLoss(label_smoothing=0.1, reduction='none')
    local_loss_fn = densenn.LocalLossMean(p.model.hidden_layer_width, nonlinearity_loss=p.model.implicit_homeostatic_loss)

    log_epochs = 10
    progress_bar = tqdm(range(1,1+(EPOCHS)))
    
    for epoch in progress_bar:
        train_epoch(model, loaders, loss_fn, local_loss_fn, opt, p, scaler, epoch)
        if epoch%1==0:
            eval_model(epoch, model, loaders, loss_fn_sum, local_loss_fn, p)          
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


