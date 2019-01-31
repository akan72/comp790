import torch
import torchvision
from torch import nn 
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import stackOverflowDataset
import networkx as nx


num_epochs = 10
batch_size = 128
learning_rate = 1e-3 

dataset = stackOverflowDataset(path="../data/mathOverflow/sx-mathoverflow.txt")

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear
        )