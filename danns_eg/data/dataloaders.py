#from data.imagenet_ffcv import ImagenetFfcvDataModule, IMAGENET_MEAN
from danns_eg.data.cifar import get_cifar_dataloaders
from danns_eg.data.mnist import (
    get_sparse_permutation_invariant_mnist_dataloaders, 
    get_sparse_permutation_invariant_fashionmnist_dataloaders)


class ToCudaTransform:
    def __call__(self, x):
        return x.to('cuda')

def get_dataloaders(p):
    if p.train.dataset == "cifar": return get_cifar_dataloaders(p)
    elif "fashion" in p.train.dataset: return get_sparse_permutation_invariant_fashionmnist_dataloaders(p, permutation_invariant=False)
    elif "perm_invariant_mnist" in p.train.dataset : return get_sparse_permutation_invariant_mnist_dataloaders(p, permutation_invariant=True)
    elif "mnist" in p.train.dataset : return get_sparse_permutation_invariant_mnist_dataloaders(p, permutation_invariant=False)
    else:print(f"ERROR: {p.train.dataset} not recognised as a vaild dataset")