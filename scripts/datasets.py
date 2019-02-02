from __future__ import print_function, division
import torch
import numpy as np
import pandas as pd 
import networkx as nx

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Stackoverflow Math Dataset
 
# SRC: id of the source node (a user)
# TGT: id of the target node (a user)
# UNIXTS: Unix timestamp (seconds since the epoch)

# TODO: Create adjacency matrix version

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


class stackOverflowDatasetWithEdges(stackOverflowDataset):
    def __getitem__(self, idx):
        # fromNode = self.data.iloc[idx, 0]
        # toNode = self.data.iloc[idx, 1:].values

        # # Node -> (Node, Time)
        # row = {
        #     'fromNode' : fromNode,
        #     'toNode' : toNode
        #     }

        # # (Node, Node) -> Time
        # row = {
        #     'fromNode': (fromNode, toNode[0]),
        #     'toNode' : toNode[1]
        # }

        # Tuple
        fromNode = self.data.iloc[idx, 0]
        toNode = self.data.iloc[idx, 1]
        timestamp = self.data.iloc[idx, 2]

        row = (fromNode, toNode, timestamp)
        

        # TODO: Apply transformation to timestamp

        if self.transform:
            row = self.transform(row)

        return row


class stackOverflowAdjacencyMatrix(stackOverflowDataset):
    def __init__(self, path, nrows, graphType=nx.DiGraph, transform=None):
        # self.data = pd.read_csv(path, sep=" ", header=None).iloc[:nrows, :2].values
        self.data = pd.read_csv(path, sep=" ", header=None)
        self.transform = transform

        graph = nx.from_edgelist(self.data.iloc[:nrows, :2].values, create_using=graphType)
        # graph = nx.from_edgelist(self.data[:nrows], create_using=graphType)

        self.adjacency = pd.DataFrame(nx.to_numpy_matrix(graph))
   
    def __len__(self):
        return len(self.adjacency)

    def __getitem__(self, idx):
        row = self.adjacency.iloc[idx, :].values

        if self.transform:
            row = self.transform(row)

        return row


# df = stackOverflowDataset(path="data/mathOverflow/sx-mathoverflow.txt")
# df = stackOverflowDatasetWithEdges(path="data/mathOverflow/sx-mathoverflow.txt")
# df = stackOverflowAdjacencyMatrix(path="data/mathOverflow/sx-mathoverflow.txt", nrows=450)
# # print(len(df[0]))
# print(df.data.shape)
