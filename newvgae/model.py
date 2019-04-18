# TODO: Try with MNIST
# TODO: Graph edit distance?
# TODO: Fix accuracy metric with decode_indices

# TODO: Try L2 loss by comparing adj_original with modle.decode(z)
# TODO: Use model.negative_sampling to get neg edge indices 
# TODO: Graph kernel similarity in loss
# TODO: Look at embeddings and reconstruction
# TODO: Play with latent code size 

# TODO: Check whether we are storing the full adjacency matrix within memory multiple times
# TODO: Step through Code 

import os.path as osp
import sys
import argparse
import pickle as pkl

from collections import defaultdict

import numpy as np
import networkx as nx

import torch
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv

from vgae import GAE, VGAE, negative_sampling
from utils import get_adjacency, plot_results, kernel_similarity, graph_edit_distance

# Define forward function of our VGAE Model 
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)

        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logvar = GCNConv(
            2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)

def main(args, kwargs):
    dataset = args.dataset.upper()

    if dataset in ['CORA', 'CITESEER', 'PUBMED']:
        path = '../data/geometric/' + dataset
        print('Using {} dataset'.format(dataset))

        dataset = Planetoid(path, dataset, T.NormalizeFeatures())
    else:
        sys.exit("You must choose one of the 'Planetoid' datasets (CORA, CITESEER, or PUBMED).")

    data = dataset[0]

    # Store the original adjacnecy matrix (for later calculation of edge prediction accuracy)
    adj_original = get_adjacency(data)

    channels = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = kwargs[args.model](Encoder(dataset.num_features, channels)).to(device)

    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data = model.split_edges(data)

    x, edge_index = data.x.to(device), data.edge_index.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Create defaultdict of lists to be used for keeping running tally of accuracy, AUC, and AP scores
    results = defaultdict(list)

    def train():
        model.train()
        optimizer.zero_grad()

        # Produces N * channels vector         
        z = model.encode(x, edge_index)    

        ''' 
        Testing MSE
        '''
        
        # TODO: Get mse here 
        # print(model.get_mse(z, adj_original))

        '''
        END TESTING
        '''


        '''
        TESTING DECODE_INDICES 
        '''
        # pos_decoded = model.decode_indices(z, data.train_pos_edge_index)

        # print('z shape: ', z.detach().numpy().shape)
        # print('pos edge index shape: ', data.train_pos_edge_index.detach().numpy().shape)

        # print('decode_indices_pos: ', pos_decoded.detach().numpy())
        # print('decode_indices_pos shape: ', pos_decoded.detach().numpy().shape)
        
        # neg_edge_index = negative_sampling(data.train_pos_edge_index, z.size(0))
        # neg_decoded = model.decode_indices(z, neg_edge_index)

        # print('decode_indices_neg: ', neg_decoded.detach().numpy())
        # print('decode_indices_neg shape: ', neg_decoded.detach().numpy().shape) 
        '''
        END TESTING
        '''

        '''
        Testing Kernel Similarity
        '''
        # g1 = nx.from_scipy_sparse_matrix(adj_original)
        # print(g1)
        
        # from sklearn.metrics.pairwise import rbf_kernel

        # adj_reconstructed = model.decode(z, sigmoid=True).detach().numpy()
        # adj_reconstructed = (adj_reconstructed > .5).astype(int)

        # similarity = kernel_similarity(adj_original, adj_reconstructed, rbf_kernel)
        # print(similarity)
        
        '''
        END TESTING
        '''



        loss = model.recon_loss(z, data.train_pos_edge_index)
        loss = loss + 0.001 * model.kl_loss()

        # TODO: Normalize epoch loss with (2/ N*N) ?
        results['loss'].append(loss)
        
        loss.backward()
        optimizer.step()

    def validate(pos_edge_index, neg_edge_index):
        model.eval()
        with torch.no_grad():
            z = model.encode(x, edge_index)

        # accuracy = model.get_accuracy(z, pos_edge_index, neg_edge_index, adj_original)
        # accuracy = model.get_accuracy_new(z, adj_original)
        auc, ap = model.test(z, pos_edge_index, neg_edge_index)

        # return accuracy, auc, ap
        return auc, ap

    def test(pos_edge_index, neg_edge_index):
        model.eval()
        with torch.no_grad():
            z = model.encode(x, edge_index)

        # accuracy = model.get_accuracy(z, pos_edge_index, neg_edge_index, adj_original)
        # accuracy = model.get_accuracy_new(z, adj_original)
        auc, ap = model.test(z, pos_edge_index, neg_edge_index)
            
        # return accuracy, auc, ap
        return auc, ap

    for epoch in range(1, args.num_epochs):
        train()

        # Run on validation edges
        # accuracy, auc, ap = validate(data.val_pos_edge_index, data.val_neg_edge_index)
        auc, ap = validate(data.val_pos_edge_index, data.val_neg_edge_index)

        # results['acc_val'].append(accuracy)
        results['auc_val'].append(auc)
        results['ap_val'].append(ap)

        # Print AUC and AP on validation data epochs
        if epoch % 10 == 0:
            # print('Val Epoch : {:03d}, ACC: {:.4f}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, accuracy, auc, ap))
            print('Val Epoch : {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))

        # Evalulate on heldout test edges for every epoch which is a multiple of the test_freq argument
        if epoch % args.test_freq == 0:
            # accuracy, auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
            auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)

            # results['acc_test'].append(accuracy)
            results['auc_test'].append(auc)
            results['ap_test'].append(ap)

            # print('Test Epoch: {:03d}, ACC: {:.4f}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, accuracy, auc, ap))
            print('Test Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))

    # Evaluate on held-out test edges 
    # auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    # print('Test AUC: {:.4f}, Test AP: {:.4f}'.format(auc, ap))

    # Pickle results 

    if args.save:    
        modelPath = args.dataset + '_RESULTS.p'
        plotPath = '../figures/geometric/' + args.dataset + '_RESULTS.png'

        pkl.dump(results, open(modelPath, 'wb'))
        plot_results(pkl.load(open(modelPath, 'rb')), path=plotPath)

        # plot_results(pkl.load(open('CORA_RESULTS.p', 'rb')), path='../figures/geometric/CORA_RESULTS.png')
        # plot_results(pkl.load(open('CITESEER_RESULTS.p', 'rb')), path='../figures/geometric/CITESEER_RESULTS.png')
        # plot_results(pkl.load(open('PUBMED_RESULTS.p', 'rb')), path='../figures/geometric/PUBMED_RESULTS.png')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='VGAE', help='Type of model (by default the base VGAE)')
    parser.add_argument('--dataset', type=str, default='CORA', help='PyTorch Geometric-Loaded Dataset')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--test_freq', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--save', type=int, default=1)

    # add arg for epochs

    args, unknown = parser.parse_known_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    assert args.model in ['GAE', 'VGAE']
    kwargs = {'GAE': GAE, 'VGAE': VGAE}

    main(args, kwargs)

