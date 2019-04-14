
# TODO: Modify torch_geo code to get accuracy metric
# TODO: Get metrics for testing 
# TODO: Step through Code 
# TODO: Look at embeddings and reconstruction
# TODO: Graph kernel similarity in loss

import os.path as osp
import sys
import argparse
from collections import defaultdict

import numpy as np
import pickle as pkl

import torch
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv

from vgae import GAE, VGAE

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
        
        z = model.encode(x, edge_index)
        
        loss = model.recon_loss(z, data.train_pos_edge_index)
        loss = loss + 0.001 * model.kl_loss()

        # TODO: Normalize epoch loss with (2/ N*N) ?
        results['loss'].append(loss)
        
        loss.backward()
        optimizer.step()


    def test(pos_edge_index, neg_edge_index):
        model.eval()
        with torch.no_grad():
            z = model.encode(x, edge_index)
            
        return model.test(z, pos_edge_index, neg_edge_index)

    for epoch in range(1, 201):
        train()
        auc, ap = test(data.val_pos_edge_index, data.val_neg_edge_index)

        results['auc_train'].append(auc)
        results['ap_train'].append(ap)

        # Print AUC and AP during training epochs
        if epoch % 10 == 0:
            print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))


    # Evaluate on held-out test edges 
    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    print('Test AUC: {:.4f}, Test AP: {:.4f}'.format(auc, ap))

    # Pickle results 
    pkl.dump(results, open('results.p', 'wb'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='VGAE', help='Type of model (by default the base VGAE)')
    parser.add_argument('--dataset', type=str, default='Cora', help='PyTorch Geometric-Loaded Dataset')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--test_freq', type=int, def=10)

    args, unknown = parser.parse_known_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    assert args.model in ['GAE', 'VGAE']
    kwargs = {'GAE': GAE, 'VGAE': VGAE}

    main(args, kwargs)

