import pickle as pkl
import numpy as np
import math
import scipy.sparse as sp
import torch
import networkx as nx
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# from torch_geometric.data import DataLoader
# from torch_geometric.datasets import MNISTSuperpixels, Planetoid
# import torch_geometric.transforms as T
from collections import defaultdict

# Get the original adjacency matrix from our PyTorch Geometric data class
def get_adjacency(dataset):
    edgeList = np.array(dataset['edge_index'].transpose(1, 0))
    edgeList = list(map(tuple, edgeList))

    d = defaultdict(list)

    for k, v in edgeList:
        d[k].append(v)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(d))

    return adj

# Get link prediction accuracy of our VGAE model
def get_accuracy(pos_edge_index, neg_edge_index, adj_original):

    model.eval()
    with torch.no_grad():
        z = model.encode(x, edge_index)

    def sigomid(x):
        return 1 / (1 + np.exp(-x))

    z = z.data.numpy()
    reconstruction = np.dot(z, z.T)
    preds_pos = []
    pos_orig = []

    for e in pos_edge_index:
        preds.append(sigomid(reconstruction[e[0], e[1]]))
        pos.append(adj_original[e[0], e[1]])

    preds_neg = []
    neg_orig = []

    for e in neg_edge_index:
        preds_neg.append(sigmoid(reconstruction[e[0], e[1]]))
        neg_orig.append(adj_original[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])

    accuracy = accuracy_score((preds_all > 0.5).astype(float), labels_all)

    return accuracy

def plot_results(results, path):
    plt.close('all')
    fig = plt.figure(figsize=(8, 8))

    # Ploting Training Loss 
    trainingLoss = results['loss']
    x_axis_train = range(len(trainingLoss))
    x_axis_test = list(range(len(results['auc_test'])))

    testfreq = math.floor(len(results['loss']) / len(results['auc_test'])) 

    x_axis_test = [x * testfreq for x in x_axis_test]

    ax = fig.add_subplot(2, 2, 1)
    ax.plot(x_axis_train, trainingLoss)
    ax.set_ylabel('ELBO Loss')
    ax.set_title('Training ELBO Loss (with KL Regularization)')
    ax.legend(['Train'], loc='upper right')

    # Plotting Training AUC 
    trainingAUC = results['auc_val']
    testingAUC = results['auc_test']

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(x_axis_train, trainingAUC)
    ax.plot(x_axis_test, testingAUC)
    ax.set_ylabel('AUC')
    ax.set_title('Training AUC')
    ax.legend(['Train', 'Test'], loc='upper right')

    # Plotting Training PC 
    trainingAP = results['ap_val']
    testingAP = results['ap_test']

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(x_axis_train, trainingAP)
    ax.plot(x_axis_test, testingAP)
    ax.set_ylabel('AP')
    ax.set_title('Training AP')
    ax.legend(['Train', 'Test'], loc='upper right')

    fig.tight_layout()
    fig.savefig(path)

plot_results(pkl.load(open('CORA_RESULTS.p', 'rb')), path='../figures/geometric/CORA_RESULTS.png')
plot_results(pkl.load(open('CITESEER_RESULTS.p', 'rb')), path='../figures/geometric/CITESEER_RESULTS.png')
plot_results(pkl.load(open('PUBMED_RESULTS.p', 'rb')), path='../figures/geometric/PUBMED_RESULTS.png')
