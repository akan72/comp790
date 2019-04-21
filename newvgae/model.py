# TODO: Try L2 loss with proper labeling
# TODO: Check whether we are storing the full adjacency matrix within memory multiple times

# TODO: Graph edit distance? try nx first

# TODO: Visualize encoding space?
# TODO: Play with latent code size 
# TODO: Sampling codes

# TODO: EM
# TODO: Differentiable graph kernel

# TODO: Fix accuracy metric with decode_indices
# TODO: Look at embeddings and reconstruction

# TODO: Use model.negative_sampling to get neg edge indices 

import os.path as osp
import pickle as pkl
import sys
from collections import defaultdict

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import MNISTSuperpixels, Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.read.planetoid import edge_index_from_dict
from urllib3 import request

# import parser
from utils import (get_adjacency, graph_edit_distance, kernel_similarity,
                   parameter_parser, plot_results)
from vgae import GAE, VGAE, negative_sampling


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
    dataset = args.data.upper()

    if dataset in ['CORA', 'CITESEER', 'PUBMED']:
        path = '../data/geometric/' + dataset
        print('Using {} dataset'.format(dataset))
        dataset = Planetoid(path, dataset, T.NormalizeFeatures())
    elif dataset == 'MNIST':
        path = '../data/geometric/' + dataset
        print('Using {} dataset'.format(dataset))
        dataset = MNISTSuperpixels(path, dataset, T.NormalizeFeatures())
    else:
        sys.exit("You must choose one of the 'Planetoid' datasets (CORA, CITESEER, or PUBMED).")

    data = dataset[0]

    # Store the original adjacnecy matrix (for later calculation of edge prediction accuracy)
    adj_original = torch.tensor(np.asarray(get_adjacency(data).toarray(), dtype=np.float32), requires_grad=False)
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
        print("in train")
        print("x", x.shape, type(x))
        print(x)
        print("edge_index", edge_index.shape, type(edge_index))
        print(edge_index)
        print("data.train_pos_edge_index", data.train_pos_edge_index.shape)
        
        input("press enter to continue...")
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

        if args.loss == 'bce':
            loss = model.recon_loss(z, data.train_pos_edge_index)
            loss = loss + 0.001 * model.kl_loss()
        elif args.loss == 'newbce':
            thing = [tuple(x) for x in data.train_pos_edge_index.numpy().transpose().tolist()]
            train_adj_original = np.asarray(nx.adjacency_matrix(nx.from_edgelist(thing)).toarray(), dtype=np.float32)

            numNodes = train_adj_original.shape[0]
            print("numNodes", numNodes)

            # print("train_adj_original",train_adj_original.shape)
            # print(train_adj_original)
            loss = model.new_recon_loss(z, train_adj_original, num_nodes=numNodes, num_channels=channels, adj_original=adj_original)
            loss = loss + 0.001 * model.kl_loss()

            print('loss: ', loss)

        elif args.loss == 'l2':
            loss = model.recon_loss_l2(z, adj_original)
            loss = loss + 0.001 * model.kl_loss()
        elif args.loss == 'anneal':
            pass 

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
        # print(accuracy)
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

    for epoch in range(1, args.epochs):
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
        if args.notes is None:
            modelPath = '../models/' + args.data + '_RESULTS.p'
            plotPath = '../figures/geometric/' + args.data + '_RESULTS.png'
        else: 
            modelPath = '../models/' + args.notes + '_' + args.data + '_RESULTS.p'
            plotPath = '../figures/geometric/' + args.notes + '_' + args.data + '_RESULTS.png'

        pkl.dump(results, open(modelPath, 'wb'))
        plot_results(pkl.load(open(modelPath, 'rb')), path=plotPath, loss=args.loss)

if __name__ == '__main__':
    args = parameter_parser()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    assert args.model in ['GAE', 'VGAE']
    kwargs = {'GAE': GAE, 'VGAE': VGAE}

    main(args, kwargs)
