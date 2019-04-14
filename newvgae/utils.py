import pickle as pkl
import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import matplotlib.pyplot as plt

# from torch_geometric.data import DataLoader
# from torch_geometric.datasets import MNISTSuperpixels, Planetoid
# import torch_geometric.transforms as T
from collections import defaultdict

def plot_results(results, path):
    plt.close('all')
    fig = plt.figure(figsize=(8, 8))

    # Ploting Training Loss 
    trainingLoss = results['loss']
    x_axis_train = range(len(trainingLoss))

    ax = fig.add_subplot(2, 2, 1)
    ax.plot(x_axis_train, trainingLoss)
    ax.set_ylabel('ELBO Loss')
    ax.set_title('Training ELBO Loss (with KL Regularization)')
    ax.legend(['Train'], loc='upper right')

    # Plotting Training AUC 
    trainingAUC = results['auc_train']

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(x_axis_train, trainingAUC)
    ax.set_ylabel('AUC')
    ax.set_title('Training AUC')
    ax.legend(['Train'], loc='upper right')

    # Plotting Training PC 
    trainingAP = results['ap_train']

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(x_axis_train, trainingAP)
    ax.set_ylabel('AP')
    ax.set_title('Training AP')
    ax.legend(['Train'], loc='upper right')

    fig.tight_layout()
    fig.savefig(path)



results = pkl.load(open('results.p', 'rb'))

plot_results(results, path='../figures/geometric/PUBMED_RESULTS.png')

