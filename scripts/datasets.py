from __future__ import print_function, division
import torch
import numpy as np
import pandas as pd 

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Stackoverflow math 
# SRC: id of the source node (a user)
# TGT: id of the target node (a user)
# UNIXTS: Unix timestamp (seconds since the epoch)

class stackOverflowDataset(Dataset):
    def __init__(self, path, transform=None): 
        self.data = pd.read_csv(path, sep=" ", header=None)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx, :2].values
        
        if self.transform:
            row = self.transform(row)

        return row