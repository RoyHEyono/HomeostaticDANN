import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
import torch
from utils import ood_scatter_plot, recon_var_plot
import danns_eg.densenet as densenets


if __name__ == '__main__':
    model_state_dict = torch.load('/home/mila/r/roy.eyono/danns_eg/dense_mnist_task/model.pth')

    input_dim = 784
    num_class = 10
    width=500
    model = densenets.DenseDANN(input_dim, width, num_class)
    model.load_state_dict(model_state_dict)

    recon_var_plot(model.cuda(), plot_variance=True)