from __future__ import print_function, division

import networkx as nx
from graphviz import Digraph

import torch
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from datasets import dynamicDataset

from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

def makeAdjLists(datapath, outpath):
    df = pd.read_csv(datapath, sep=" ", header=None)

    df[3] = pd.to_datetime(df[2], unit='s')
    df.columns = ['from', 'to', 'unixTime', 'timestamp']
    df.set_index('timestamp', inplace=True)

    data = df.groupby(by=[df.index.year, df.index.month])
    dfs = []
                
    for key in list(data.groups.keys()):
        dfs.append(data.get_group(key))

    combined = [dfs[0]]
    for i in range(1, len(dfs)):
        combined.append(pd.concat([combined[i-1], dfs[i]]))

    total = 0
    for i in range(len(combined)):
        total += len(dfs[i])
        assert(len(combined[i]) == total)

    for i in range(len(combined)):
        adjacency = nx.from_edgelist(combined[i].iloc[:, :2].values, create_using=nx.DiGraph)
        nx.write_adjlist(adjacency, outpath + str(i) + ".adjlist")


# makeAdjLists("../data/mathOverflow/sx-mathoverflow.txt", "../data/temp/edgeLists/test")
