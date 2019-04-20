import argparse
import pickle as pkl
import numpy as np
import math
import scipy.sparse as sp
import torch
import networkx as nx
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel, cosine_similarity, laplacian_kernel
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

def plot_results(results, path, loss):
    plt.close('all')
    fig = plt.figure(figsize=(8, 8))

    # Ploting Training Loss 
    trainingLoss = results['loss']
    x_axis_train = np.array(range(len(trainingLoss)))

    testfreq = math.floor(len(results['loss']) / len(results['auc_test'])) 

    x_axis_test = np.array(list(range(len(results['auc_test']))))
    x_axis_test = [x * testfreq for x in x_axis_test]

    ax = fig.add_subplot(3, 1, 1)

    ax.plot(x_axis_train, trainingLoss)
    ax.set_ylabel('ELBO Loss')
    ax.set_xlabel('Epoch')
    ax.set_title('Training ELBO Loss (' + loss.upper() + ' reconstruction)')
    ax.legend(['Train'], loc='upper right')

    # Plotting Accuracy

    # trainingACC = np.array(results['acc_val'])
    # testingACC = np.array(results['acc_test'])

    # print(trainingLoss.shape)
    # print(trainingACC.shape)

    # ax = fig.add_subplot(2, 2, 2)
    # ax.plot(x_axis_train, trainingACC)
    # ax.plot(x_axis_test, testingACC)
    # ax.set_ylabel('Accuracy')
    # ax.set_title('Model Accuracy')
    # ax.legend(['Train', 'Test'], loc='upper right')

    # Plotting AUC 
    trainingAUC = results['auc_val']
    testingAUC = results['auc_test']

    ax = fig.add_subplot(3, 1, 2)
    ax.plot(x_axis_train, trainingAUC)
    ax.plot(x_axis_test, testingAUC)
    ax.set_ylabel('AUC')
    ax.set_xlabel('Epoch')
    ax.set_title('Model AUC')
    ax.legend(['Train', 'Test'], loc='upper right')

    # Plotting AP 
    trainingAP = results['ap_val']
    testingAP = results['ap_test']

    ax = fig.add_subplot(3, 1, 3)
    ax.plot(x_axis_train, trainingAP)
    ax.plot(x_axis_test, testingAP)
    ax.set_ylabel('AP')
    ax.set_xlabel('Epoch')
    ax.set_title('Model AP')
    ax.legend(['Train', 'Test'], loc='upper right')

    fig.tight_layout()
    fig.savefig(path)

def kernel_similarity(original , reconstructed, kernel): 

    G1 = nx.from_numpy_matrix(original)
    G2 = nx.from_numpy_matrix(reconstructed)

    assert(len(G1) == len(G2))

    t1 = nx.floyd_warshall(original)
    t2 = nx.floyd_warshall(reconstructed)

    numNodes = len(G1)
    t1paths = []
    t2paths = []

    for i in range(numNodes):
        t1paths.append(list(t1[i].values()))
        t2paths.append(list(t2[i].values()))

    kernelResult = sum(sum(kernel(t1paths, t2paths)))
    print(kernelResult)

    return kernelResult

def graph_edit_distance(original, reconstructed):
    
    G1 = nx.from_scipy_sparse_matrix(original)
    G2 = nx.from_numpy_matrix(reconstructed)

    edit_distance = nx.algorithms.summary.optimize_graph_edit_distance(G1, G2)
    print(edit_distance)

    return edit_distance


def parameter_parser():
    parser = argparse.ArgumentParser(description= 'VGAE')

    parser.add_argument('--data',
                        type=str,
                        default='CORA',
                        help='PyTorch Geometric-Loaded Dataset.')
    
    parser.add_argument('--save',
                        type=int,
                        default=1,
                        help='Whether to save current results, 1==Yes, 0==No.')
                        
    parser.add_argument('--notes',
                        type=str,
                        default=None,
                        help='Add description to pickled results and output plots.')

    parser.add_argument('--loss',
                        type=str,
                        choices=['bce', 'newbce', 'l2', 'anneal'],
                        default='bce',
                        help='Loss type.')

    parser.add_argument('--model', 
                        type=str,
                        default='VGAE',
                        help='Type of model (by default the base VGAE)')

    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='Random seed.')

    parser.add_argument('--test_freq',
                        type=int,
                        default=10,
                        help='Frequency of (measured in epochs).')

    parser.add_argument('--epochs',
                        type=int,
                        default=200,
                        help='Number of epochs for which the model is run.')

    return parser.parse_args()
