import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
import torch_geometric.utils as torch_util
import torch_scatter
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MNISTSuperpixels, Planetoid
from torch_geometric.nn import ChebConv, GCNConv, SAGEConv

def get_data():
    dataset = 'Mnist'
    path = '../data/geometric/MNIST'
    trainset = MNISTSuperpixels(root=path, train=True)
    testset = MNISTSuperpixels(root=path, train=False)

    lenTrain = len(trainset)
    lenTest = len(testset)

    trainLoader = DataLoader(trainset[:lenTrain//125], batch_size=1, shuffle=False)
    testloader = DataLoader(testset[:lenTest//125], batch_size=1,  shuffle=False)
    
    return trainLoader, testloader


train, test = get_data()

print(len(train), len(test))

# for step, batch in enumerate(train):
#     x = batch['x']
#     adj = batch['edge_index']

#     print(x.shape, type(x))
#     print(adj.shape, type(adj))
