import os
from tqdm import tqdm
from typing import Optional, Mapping
from pprint import pprint
from dataclasses import dataclass, field

import numpy as np
from omegaconf import II, MISSING, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import torch.backends.xnnpack
print("XNNPACK is enabled: ", torch.backends.xnnpack.enabled, "\n")

from danns_eg import utils as danns_eg_utils
from danns_eg import dense
from danns_eg import optimisation
from danns_eg.sequential import Sequential
from danns_eg.data import r_of_k

import train_utils

def build_dataloaders(cfg: Mapping):
    X,y, w_star = r_of_k.generate_noisy_r_of_k_data(
                                            n_datapoints=cfg.dataset.n_train_datapoints,
                                            n=cfg.dataset.n, 
                                            k=cfg.dataset.k, 
                                            r=0, 
                                            neg_wstar=True, 
                                            verbose=True,
                                            noise_probs=(0.4, 0.2, 0.4))
    
    X_val, y_val, _ = r_of_k.generate_noisy_r_of_k_data(
                                            n_datapoints=cfg.dataset.n_val_datapoints,
                                            n=cfg.dataset.n, 
                                            k=cfg.dataset.k, 
                                            r=0, 
                                            w_star=w_star,
                                            verbose=True,
                                            noise_probs=None)
    

    X_test, y_test, _ = r_of_k.generate_noisy_r_of_k_data(
                                            n_datapoints=cfg.dataset.n_test_datapoints,
                                            n=cfg.dataset.n, 
                                            k=cfg.dataset.k, 
                                            r=0, 
                                            w_star=w_star,
                                            verbose=True,
                                            noise_probs=None)
    
    loaders = {}
    for split in ['train', 'val', 'test']:
        if split == 'train':
            rofk_dataset = train_utils.Dataset(X, y) 
            
        elif split == 'val':
            rofk_dataset = train_utils.Dataset(X_val, y_val) 
            
        elif split == 'test':
            rofk_dataset = train_utils.Dataset(X_test, y_test) 

        loaders[split] = DataLoader(rofk_dataset,
                                    batch_size=cfg.dataset.batch_size,
                                    shuffle=True if 'train' in split else False)
    return loaders


def build_model(cfg):
    """
    For r-of-k we are using single neuron dense layers with no bias and
    identity act func (because the bineary CE loss takes logit inputs). 
    W is also init as zeros.
    """
    # assert cfg.update_algorithm in ["eg", "gd"]
    # if cfg.update_algorithm == "eg": split_bias = True
    # elif cfg.update_algorithm == "gd": split_bias = False
    # assert cfg.model in ["dann", "mlp"]
    if cfg.model.name == "dann":
        model = dense.EiDenseLayer(n_input=cfg.dataset.n, ne=1, ni=1, use_bias=False, nonlinearity=None)
        model.patch_init_weights_method(train_utils.init_eidense_rofk)
        model.init_weights()
    
    elif cfg.model.name == "mlp":
        model = dense.DenseLayer(n_input=cfg.dataset.n, n_output=1, use_bias=False, nonlinearity=None)
        model.patch_init_weights_method(train_utils.init_dense_rofk)
        model.init_weights()

    return model

def get_optimiser(model: torch.nn.Module, cfg: Mapping, verbose=False):
    """
    Groups that may be returned by opt.get_param_groups(model) for rofk are 
    ["wix_params", "wei_params",'wex_params', 'other_params', 'biases']
    """
    param_groups_list = optimisation.get_param_groups(model)
    for param_group in param_groups_list:

        group_name = param_group["name"]
        if group_name in ["wix_params", "wei_params",'wex_params']: 
            param_group["positive_only"] = True

        if cfg.model.csgd: # here we can hardcode as is only one layer
            if group_name == "wix_params": 
                param_group['lr'] = cfg.opt.lr # /np.sqrt(ne) which is 1

            elif group_name == "wei_params": 
                param_group['lr'] = cfg.opt.lr / cfg.dataset.n
        
        elif not cfg.model.csgd:
            if group_name == "wix_params":
                if cfg.model.opt.wix_lr != "wex_lr": 
                    param_group['lr'] = cfg.model.opt.wix_lr 
            elif group_name == "wei_params": 
                if  cfg.model.opt.wei_lr != "wex_lr": 
                    param_group['lr'] = cfg.model.opt.wei_lr 

    opt = optimisation.SGD(params = param_groups_list, 
                           lr = cfg.opt.lr,
                           weight_decay = cfg.opt.wd,
                           momentum = cfg.opt.momentum,
                           update_algorithm = cfg.opt.update_algorithm) 
    return opt

def train_epoch(cfg, opt, model, loaders, epoch_i, scaler): 
    """
    Returns batch_accs, batch_losses lists
    """
    
    loss_func = torch.nn.functional.binary_cross_entropy_with_logits
    
    batch_accs, batch_losses = [], []
    progress_bar = tqdm(loaders['train'], desc='Train')
    for batch_i, (X, y) in enumerate(progress_bar):
        X = X.to(device)
        y = y.to(device)

        model.train()
        opt.zero_grad(set_to_none=True)
        with autocast():
            logits = model(X)
            loss = loss_func(logits.squeeze(), y.float())
        
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            y_hat = (probs>0.5).long() # labels are long (int 32)
            batch_acc = train_utils.binary_acc(y_hat, y)
        progress_bar.set_description(
            f'Epoch {epoch_i}, batch {batch_i+1}: Acc {np.mean(batch_accs)*100}% '
        )
        batch_accs.append(batch_acc.item())
        batch_losses.append(loss.item())

    # print("Accs:", batch_accs)
    # print("Losses:", batch_losses)
    return batch_losses, batch_accs

def eval_model(results_dict, model, loaders, epoch_i):
    """
    This func assumes that all batch sizes are equal
    """
    loss_func = torch.nn.functional.binary_cross_entropy_with_logits
    model.eval()
    #with autocast:
    with torch.no_grad():
        for key in ["train", "val", "test"]:
            loss, acc, n = 0,0,0
            for X, y in loaders[key]:
                X = X.to(device)
                y = y.to(device)
                logits = model(X)
                probs = torch.sigmoid(logits)
                y_hat = (probs>0.5).long() # labels are long (int 32)
                loss += loss_func(logits.squeeze(), y.float()).item()
                acc += train_utils.binary_acc(y_hat, y).item()*100
                n += y.size(0)
            results_dict[key+"_losses"].append(loss/n)
            results_dict[key+"_accs"].append(acc/n)

@hydra.main(config_path = "conf", config_name = "main_config", version_base=None)
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    
    model = build_model(cfg)
    model.to(device)
    
    opt = get_optimiser(model, cfg)
    print(opt)

    loaders = build_dataloaders(cfg)
    scaler = GradScaler() 

    results_dict = {
        'train_losses': [], 'train_accs': [],
        'val_losses': [], 'val_accs': [],
        'test_losses': [], 'test_accs': [],
    }
    for epoch_i in range(cfg.n_epochs):
        eval_model(results_dict=results_dict, model=model, 
                   loaders=loaders, epoch_i=epoch_i)
        train_epoch(cfg, opt=opt, model=model, loaders=loaders, 
                    epoch_i=epoch_i, scaler=scaler)
        
    pprint(results_dict)

    

if __name__ == "__main__":
    device = danns_eg_utils.get_device()
    main()