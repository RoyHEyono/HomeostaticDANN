import os
import hydra
from typing import Optional
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from pprint import pprint
from dataclasses import dataclass, field

import torch
from typing import Mapping
from torch.utils.data import DataLoader
from omegaconf import II, MISSING, OmegaConf

from danns_eg import dense
from danns_eg import optimisation
from danns_eg.sequential import Sequential
from danns_eg.data import r_of_k

import train_utils

def build_dataloaders(cfg: Mapping):
    X,y, w_star = r_of_k.generate_noisy_r_of_k_data(
                                            n_datapoints=cfg.dataset.n_datapoints,
                                            n=cfg.dataset.n, 
                                            k=cfg.dataset.k, 
                                            r=0, 
                                            neg_wstar=True, 
                                            verbose=True,
                                            noise_probs=(0.4, 0.2, 0.4))
    
    X_test, y_test, _ = r_of_k.generate_noisy_r_of_k_data(
                                            n_datapoints=cfg.dataset.n_test_datapoints,
                                            n=cfg.dataset.n, 
                                            k=cfg.dataset.k, 
                                            r=0, 
                                            w_star=w_star,
                                            verbose=True,
                                            noise_probs=None)
    
    loaders = {}
    for split in ['train','test']:
        if split == 'train':
            rofk_dataset = train_utils.Dataset(X, y) 
            
        elif split == 'test':
            # no test set right now
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
    if cfg.name == "dann":
        model = dense.EiDenseLayer(n_input=cfg.n, ne=1, ni=1, use_bias=False, nonlinearity=None)
        model.patch_init_weights_method(train_utils.init_eidense_rofk)
        model.init_weights()
    
    elif cfg.name == "mlp":
        model = dense.DenseLayer(n_input=cfg.n, n_output=1, use_bias=False, nonlinearity=None)
        model.patch_init_weights_method(train_utils.init_dense_rofk)
        model.init_weights()

    return model

def get_optimiser(model: torch.nn.Module, cfg: Mapping):
    """
    Groups that may be returned by opt.get_param_groups(model) for rofk are 
    ["wix_params", "wei_params",'wex_params', 'other_params', 'biases']
    
    """
    param_groups_list = optimisation.get_param_groups(model)
    pprint(param_groups_list)
    for param_group in param_groups_list:
        group_name = param_group["name"]
        if group_name in ["wix_params", "wei_params",'wex_params']: 
            param_group["positive_only"] = True

        if cfg.opt.use_sep_inhib_lrs:
            if group_name == "wix_params": param_group['lr'] = cfg.opt.wix
            elif group_name == "wei_params": param_group['lr'] = cfg.opt.wei

    pprint(param_groups_list)

    opt = optimisation.SGD(params = param_groups_list, 
                           lr = cfg.opt.lr,
                           weight_decay = cfg.opt.wd,
                           momentum = cfg.opt.momentum,
                           update_algorithm = cfg.opt.update_algorithm) 
    return opt

def train_epoch(cfg: Mapping):
    pass

@hydra.main(config_path = "conf", config_name = "main_config", version_base=None)
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    exit()

    model = build_model(cfg.model)
    print(" ---- ")
    print(list(model.named_modules()))
    opt = get_optimiser(model, cfg)
    print(opt)
    print(" ---- ")

    loaders = build_dataloaders(cfg)
    

if __name__ == "__main__":
    main()