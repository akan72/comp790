from __future__ import print_function, division
import torch
import numpy as np
import pandas as pd 
import networkx as nx
import os
import pickle as pkl


from tqdm import tqdm

os.chdir("data/GraphML/span(1year(s))")
files = os.listdir()

output = []
for f in tqdm(files):
    df = nx.read_graphml(f)
    output.append(nx.to_numpy_matrix(df))


pkl.dump(output, open("span(1(year(s)).pkl", "wb"))
