#
#return the sum of the residual of the computed eigenvectors plus the sum of
#the residual of the computed eigenvalues.
#This will be the loss function to train the SpectralNet.

import torch
import torch_geometric
from torch_geometric.utils import get_laplacian


def laplacian(graph):
    lap = get_laplacian(graph.edge_index,
                        num_nodes=graph.num_nodes,
                        normalization='rw')#left normalized laplacian
    return torch.sparse_coo_tensor(lap[0],lap[1])

def rayleigh_quotient(x,L):
    return x.t().matmul(L).matmul(x)/(x.t().matmul(x))

def residual(x,L,mse):
    return mse(L.matmul(x),rayleigh_quotient(x,L)*x)+rayleigh_quotient(x,L)

def loss_spectral_embedding(x,L):
    mse = torch.nn.MSELoss()
    l = torch.tensor(0.0)
    for i in range(x.shape[1]):
        l += residual(x[:,i],L,mse)
    return l

