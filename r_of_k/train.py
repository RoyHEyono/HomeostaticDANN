import os
import hydra
from omegaconf import DictConfig
from pprint import pprint

import torch
from torch.utils.data import DataLoader, Dataset

from danns_eg import dense
from danns_eg import optimisation
from danns_eg.sequential import Sequential
from danns_eg.data import r_of_k

class Dataset(Dataset):
    """
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    Mnist with n+1 digits presented at once, 1st is true target. In this
    implementation the same distractors are presented with each true target. 
    An alternative would be to make them random. 

    Permutate data by setting idx_permute attribute
    https://discuss.pytorch.org/t/permutate-mnist-help-needed/22901
    
    """
    def __init__(self,X, y):
        """
        n is the number of distractors, so n=0 is normal mnist 
        """
        super().__init__()
        self.X = X
        self.y =y

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def init_eidense_rofk(layer):
    """
    Initialise for W vector of zeros
    """
    layer.Wex.data = torch.ones(layer.Wex.shape)*(1/layer.n_input) 
    layer.Wix.data = torch.ones(layer.Wix.shape)*(1/layer.n_input) 
    layer.Wei.data = torch.ones(layer.Wei.shape)

def init_dense_rofk(layer):
    """
    Initialise for W vector of zeros
    """
    layer.W.data = torch.zeros(layer.W.shape)

def build_model(cfg):
    """
    For r-of-k we are using single neuron dense layers with no bias and
    identity act func (because the bineary CE loss takes logit inputs). 
    W is also init as zeros.
    """
    # assert cfg.update_algorithm in ["eg", "gd"]
    # if cfg.update_algorithm == "eg": split_bias = True
    # elif cfg.update_algorithm == "gd": split_bias = False
    assert cfg.model in ["dann", "mlp"]
    if cfg.model == "dann":
        model = dense.EiDenseLayer(n_input=cfg.dataset.n, ne=1, ni=1, use_bias=False, nonlinearity=None)
        model.patch_init_weights_method(init_eidense_rofk)
        model.init_weights()
    
    elif cfg.model == "mlp":
        model = dense.DenseLayer(n_input=cfg.dataset.n, n_output=1, use_bias=False, nonlinearity=None)
        model.patch_init_weights_method(init_dense_rofk)
        model.init_weights()

    return model

def get_optimiser(model, cfg):
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


def build_dataloaders(cfg):
    X,y, w_star = r_of_k.generate_noisy_r_of_k_data(
                                            n_datapoints=cfg.dataset.n_datapoints,
                                            n=cfg.dataset.n, 
                                            k=cfg.dataset.k, 
                                            r=0, 
                                            neg_wstar=True)
    print(X.shape)
    rofk_dataset = Dataset(X, y) 
    print(rofk_dataset)
    loaders = {}
    for split in ['train','test']:
        if split == 'train':
            loaders[split] = DataLoader(rofk_dataset,
                                        batch_size=cfg.dataset.batch_size,
                                        shuffle=True)
        if split == 'test':
            # no test set right now
            loaders[split] = None
    return loaders

@hydra.main(config_path="conf", config_name="config")
def main(cfg:DictConfig):
    print(cfg)

    model = build_model(cfg)
    print(" ---- ")
    print(list(model.named_modules()))
    opt = get_optimiser(model, cfg)
    print(opt)
    
    
    
    # loaders = build_dataloaders(cfg)
    # breakpoint()

    # model = dense.EiDenseLayer(n_input=cfg.dataset.n,ne=1, ni=1, split_bias=False)
    # print(model.W)
    # model.patch_init_weights_method(init_eidense_rofk)
    # model.init_weights()
    # print(model.W)

    # model = dense.DenseLayer(n_input=cfg.dataset.n,n_output=1, split_bias=False)
    # print(model.W)
    # model.patch_init_weights_method(init_dense_rofk)
    # model.init_weights()
    # print(model.W)

    


    
    #optimizer = build_optimizer()
    # will have loop inside main
    #train_epoch(loaders,model)
    # eval

if __name__ == "__main__":
    main()