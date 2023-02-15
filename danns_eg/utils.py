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




    