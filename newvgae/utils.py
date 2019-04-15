import pickle as pkl
import numpy as np
import math
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

results = pkl.load(open('results.p', 'rb'))

plot_results(results, path='../figures/geometric/CORA_RESULTS.png')

