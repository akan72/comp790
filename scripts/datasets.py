from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd 
import networkx as nx
import glob
import dynetx as dn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Stackoverflow Math Dataset (Temporarl)
 
# SRC: id of the source node (a user)
# TGT: id of the target node (a user)
# UNIXTS: Unix timestamp (seconds since the epoch)

class stackOverflowDataset(Dataset):
    def __init__(self, path, transform=None): 
        self.data = pd.read_csv(path, sep=" ", header=None)

        # Convert unix times to human read
        self.data[3] = pd.to_datetime(self.data[2], unit='s')
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
        unixTime = self.data.iloc[idx, 2]
        timestamp = self.data.iloc[idx, 3]

        row = (fromNode, toNode, unixTime, timestamp)
        

        # TODO: Apply transformation to timestamp

        if self.transform:
            row = self.transform(row)

        return row


class stackOverflowAdjacencyMatrix(Dataset):
    def __init__(self, path, nrows=None, graphType=nx.DiGraph, transform=None):
        # self.data = pd.read_csv(path, sep=" ", header=None).iloc[:nrows, :2].values
        self.data = pd.read_csv(path, sep=" ", header=None)
        self.data[3] = pd.to_datetime(self.data[2], unit='s')
        self.transform = transform

        if nrows is None:
            nrows = self.data.shape[0]

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
# print(len(df[0]))
# print(df.data.shape)
 
class oldDynamicDataset(Dataset):
    def __init__(self, path, nrows=None, graphType=nx.DiGraph, transform=None):
        # self.data = pd.read_csv(path, sep=" ", header=None).iloc[:nrows, :2].values
        self.data = pd.read_csv(path, sep=" ", header=None)
        self.data[3] = pd.to_datetime(self.data[2], unit='s') 
        self.data.columns = ['from', 'to', 'unixTime', 'timestamp']
        self.data.set_index('timestamp', inplace=True)

        self.transform = transform

        if nrows is None:
            nrows = self.data.shape[0]

        # graph = nx.from_edgelist(self.data.iloc[:nrows, :2].values, create_using=graphType)
        # graph = nx.from_edgelist(self.data[:nrows], create_using=graphType)
        # self.adjacency = pd.DataFrame(nx.to_numpy_matrix(graph))
   
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #row = self.adjacency.iloc[idx, :].values
        row = self.data.iloc[idx, :].values
        if self.transform:
            row = self.transform(row)

        return row

# df = oldDynamicDataset(path="data/enronEmployees/ia-enron-employees/ia-enron-employees.txt")
# df = oldDynamicDataset(path="data/mathOverflow/sx-mathoverflow.txt")

# df  = dn.read_interactions(path="data/mathOverflow/sx-mathoverflow.txt", directed=True, delimiter=" ", nodetype=int, timestamptype=int)
# dn.write_interactions(df, "data/mathOverflow/dyngraph.txt")


class dynamicDataset(Dataset):
    def __init__(self, path, transform=None):
        self.data = []

        for filename in glob.glob(os.path.join(path, '*.adjlist')):
            G = nx.read_adjlist(filename)

            adjList = []
            for adj in nx.generate_adjlist(G):
                adjList.append([int(i) for i in adj.split()])

            self.data.append(adjList)

    def __len__(self):
        print(len(self.data))

    def __getitem__(self, idx):
        return(self.data[idx])
            



