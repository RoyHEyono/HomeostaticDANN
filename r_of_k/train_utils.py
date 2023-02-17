import torch
from torch.utils.data import Dataset

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
        self.X = X.astype("float32")
        self.y =y.astype("long")

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

def binary_acc(yhat, y):
    yhat = yhat.squeeze()
    n_correct = torch.sum(yhat==y)
    return 1.0*n_correct/y.size(0)