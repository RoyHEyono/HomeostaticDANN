import os
import hydra
from omegaconf import DictConfig
import torch

from danns_eg import dense
from danns_eg.data import r_of_k

from torch.utils.data import DataLoader, Dataset

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
    


def build_model(cfg):
    """
    For r-of-k we are using single neurons with linear outputs because
    the bineary CE loss will expect logit inputs 
    """
    if cfg.update_algorithm == "eg": split_bias = True
    elif cfg.update_algorithm == "gd": split_bias = False
    
    assert cfg.model in ["dann", "mlp"]
    if cfg.model == "dann":
        model = dense.EiDenseLayer(cfg.dataset.n, 2, 1, split_bias)
    elif cfg.model == "mlp":
        model = dense.DenseLayer(cfg.dataset.n, 2, split_bias)

    return model

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
    print(cfg.dataset.n)
    #print("Working directory : {}".format(os.getcwd()))

    model = build_model(cfg)
    loaders = build_dataloaders(cfg)
    breakpoint()
    
    #optimizer = build_optimizer()
    # will have loop inside main
    #train_epoch(loaders,model)
    # eval

if __name__ == "__main__":
    main()