import argparse

import time
import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
import torch_geometric.utils as torch_util
import torch_scatter

from torch_geometric.data import DataLoader
from torch_geometric.datasets import MNISTSuperpixels, Planetoid
from torch_geometric.nn import ChebConv, GCNConv, SAGEConv

from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam

import visdom
from utils.mnist_cached import MNISTCached as MNIST
from utils.mnist_cached import setup_data_loaders
from utils.vae_plots import mnist_test_tsne, plot_llk, plot_vae_samples

torch.set_default_tensor_type('torch.FloatTensor')

# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(z|x)
class Encoder(nn.Module):
    def __init__(self, n_feat, z_dim, hidden_dim):
        super(Encoder, self).__init__()

        # Set up thhe Graph convolutional layers 
        self.gc1 = GCNConv(n_feat, hidden_dim)
        self.gc2_mu = GCNConv(hidden_dim, z_dim)
        self.gc2_sig = GCNConv(hidden_dim, z_dim)
        
        # self.gc1 = SAGEConv(n_feat, hidden_dim)
        # self.gc2_mu = SAGEConv(hidden_dim, z_dim)
        # self.gc2_sig = SAGEConv(hidden_dim, z_dim)    
        
        # Setup for non-linearities
        self.softplus = nn.Softplus()
        self.relu = torch.nn.ReLU()

    def forward(self, x, adj):
        # define the forward computation on the adjacency matrix for each graph and its features, x 
    
        # x = F.relu(self.gc1(x, adj))
        # z_loc = self.gc2_mu(x, adj)
        # z_scale = torch.exp(self.gc2_sig(x, adj))

        hidden = self.softplus(self.gc1(x, adj))
        z_loc = self.gc2_mu(hidden, adj)
        z_scale = torch.exp(self.gc2_sig(hidden, adj))

        return z_loc, z_scale


# define the PyTorch module that parameterizes the
# observation likelihood p(x|z)
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Decoder, self).__init__()
        # setup the two linear transformations used for the decoder 
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, 75)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units

        hidden = self.softplus(self.fc1(z))

        # return the parameter for the output Bernoulli
        loc_img = torch.sigmoid(self.fc21(hidden))
        return loc_img


# define a PyTorch module for the VAE
class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, z_dim=50, hidden_dim=400, use_cuda=False):
        super(VAE, self).__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(1, z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    # define the model p(x|z)p(z)
    def model(self, x, adj):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc_img = self.decoder.forward(z)
            # score against actual graphs

            pyro.sample("obs", dist.Bernoulli(loc_img.reshape(-1, x.shape[0])).to_event(1), obs=x.reshape(-1, x.shape[0]))
            # return the loc so we can visualize it later
            return loc_img

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x, adj):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x, adj)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    # define a helper function for reconstructing images
    def reconstruct_graph(self, x, adj):
        # encode image x
        z_loc, z_scale = self.encoder(x, adj)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img


def main(args):
    # clear param store
    pyro.clear_param_store()

    ### SETUP
    train_loader, test_loader = get_data()

    # setup the VAE
    vae = VAE(use_cuda=args.cuda)

    # setup the optimizer
    adam_args = {"lr": args.learning_rate}
    optimizer = Adam(adam_args)

    # setup the inference algorithm
    elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
    svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)

    inputSize = 0 

    # setup visdom for visualization
    if args.visdom_flag:
        vis = visdom.Visdom()

    train_elbo = []
    test_elbo = []

    for epoch in range(args.num_epochs):
        # initialize loss accumulator
        epoch_loss = 0.
        # do a training epoch over each mini-batch x returned
        # by the data loader

        for step, batch in enumerate(train_loader):
            x, adj = 0, 0
            # if on GPU put mini-batch into CUDA memory
            if args.cuda:
                x = batch['x'].cuda()
                adj = batch['edge_index'].cuda()
            else:

                x = batch['x']
                adj = batch['edge_index']

            inputSize = x.shape[0] * x.shape[1]
            epoch_loss += svi.step(x, adj)

        # report training diagnostics
        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train
        train_elbo.append(total_epoch_loss_train)
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

        if True:
        # if epoch % args.test_frequency == 0:
            # initialize loss accumulator
            test_loss = 0.
            # compute the loss over the entire test set
            for step, batch in enumerate(test_loader):
                x, adj = 0, 0
                # if on GPU put mini-batch into CUDA memory
                if args.cuda:
                    x = batch['x'].cuda()
                    adj = batch['edge_index'].cuda()
                else:
                    x  = batch['x']
                    adj = batch['edge_index']
                # compute ELBO estimate and accumulate loss
                # print('before evaluating test loss')
                test_loss += svi.evaluate_loss(x, adj)
                # print('after evaluating test loss')

                # pick three random test images from the first mini-batch and
                # visualize how well we're reconstructing them
                # if i == 0:
                #     if args.visdom_flag:
                #         plot_vae_samples(vae, vis)
                #         reco_indices = np.random.randint(0, x.shape[0], 3)
                #         for index in reco_indices:
                #             test_img = x[index, :]
                #             reco_img = vae.reconstruct_img(test_img)
                #             vis.image(test_img.reshape(28, 28).detach().cpu().numpy(),
                #                       opts={'caption': 'test image'})
                #             vis.image(reco_img.reshape(28, 28).detach().cpu().numpy(),
                #                       opts={'caption': 'reconstructed image'})


                if args.visdom_flag:
                    plot_vae_samples(vae, vis)
                    reco_indices = np.random.randint(0, x.shape[0], 3)  
                    for index in reco_indices:
                        test_img = x[index, :]
                        reco_img = vae.reconstruct_graph(test_img)
                        vis.image(test_img.reshape(28, 28).detach().cpu().numpy(),
                                    opts={'caption': 'test image'})
                        vis.image(reco_img.reshape(28, 28).detach().cpu().numpy(),
                                    opts={'caption': 'reconstructed image'})

            # report test diagnostics
            normalizer_test = len(test_loader.dataset)
            total_epoch_loss_test = test_loss / normalizer_test
            test_elbo.append(total_epoch_loss_test)
            print("[epoch %03d]  average test loss: %.4f" % (epoch, total_epoch_loss_test))

        if epoch == args.tsne_iter:
            mnist_test_tsne(vae=vae, test_loader=test_loader)
            plot_llk(np.array(train_elbo), np.array(test_elbo))

    if args.save:
        torch.save({
            'epoch': epoch,
            'model_state_dict': vae.state_dict(),
            'optimzier_state_dict': optimizer.get_state(),
            'train_loss': total_epoch_loss_train,
            'test_loss': total_epoch_loss_test,
            'trainELBO': train_elbo,
            'testELBO': test_elbo
            }, '../models/vae_' + args.name + str(args.time) +'.pt')

    return vae


def get_data():
    dataset = args.name
    path = '../data/geometric/MNIST'
    trainset = MNISTSuperpixels(path, train=True)
    testset = MNISTSuperpixels(path, train=False)

    lenTrain = len(trainset)
    lenTest = len(testset)

    trainLoader = DataLoader(trainset[:lenTrain//50], batch_size=1, shuffle=False)
    testloader = DataLoader(testset[:lenTest//50], batch_size=1, shuffle=False)
    return trainLoader, testloader

if __name__ == '__main__':
    assert pyro.__version__.startswith('0.3.1')
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=10, type=int, help='number of training epochs')
    parser.add_argument('-tf', '--test-frequency', default=5, type=int, help='how often we evaluate the test set')
    parser.add_argument('-lr', '--learning-rate', default=2.0e-3, type=float, help='learning rate')
    
    parser.add_argument('--cuda', action='store_true', default=False, help='whether to use cuda')
    parser.add_argument('--jit', action='store_true', default=False, help='whether to use PyTorch jit')
    parser.add_argument('-visdom', '--visdom_flag', action="store_true", help='Whether plotting in visdom is desired')
    parser.add_argument('-i-tsne', '--tsne_iter', default=100, type=int, help='epoch when tsne visualization runs')
    parser.add_argument('--time', default=int(time.time()), help="Current system time")

    parser.add_argument('--name', default='Mnist', help="Name of the dataset")
    parser.add_argument('--save', default=True, help="Whether to save the trained model")

    args = parser.parse_args()

    model = main(args)

