from phase1net import SpectralNet
from trainingdataset import mixed_dataset
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data,Batch
from torch_geometric.data import DataLoader
from phase1loss import loss_spectral_embedding,laplacian
import os
import numpy as np


#device = 'GPU' if torch.cuda.is_available() else 'CPU'
def constructtrainingdataset(datapath):
    ng = 2000
    n_iter,n_iter_suitesparse = 15,3
    listdata,count_fem,count_suitesparse = mixed_dataset(ng,100,5000,n_iter,n_iter_suitesparse,datapath)
    trainingdataset = []
    for adj in listdata:
        row = adj.row
        col = adj.col
        rowcols = np.array([row,col])
        edges = torch.tensor(rowcols,dtype=torch.long)
        #initialize node feature vectors with random values
        nodes = torch.randn(adj.shape[0],2)
        trainingdataset.append(Data(x=nodes,edge_index=edges))
    return trainingdataset,ng,count_fem,count_suitesparse

def trainingloop(datapath,modelpath,device):
    print("Starting Spectral Embedding Module Training ")
    f = SpectralNet().to(device)
    print("Number of parameters:", sum(p.numel() for p in f.parameters()))

    #loss function
    loss_f = loss_spectral_embedding

    #training dataset
    #construct training dataset
    trainingdataset,ng,count_fem,count_suitesparse = constructtrainingdataset(datapath)
    loader = DataLoader(trainingdataset,batch_size=1,shuffle=True)
    print("Training dataset for spectral embedding module done")
    print("Number of training graphs:", len(loader))
    print("Number of Delanauy graphs:", ng+int(ng/2))
    print("Number of FEM graphs:", count_fem)
    print("Number of SuiteSparse graphs:", count_suitesparse)

    #optimizer
    init_lr = 0.0001
    optimizer = torch.optim.Adam(f.parameters(), lr=init_lr)
    epochs = 120
    losses = []
    update = torch.tensor(5).to(device)
    print_loss_every = 10

    #training loop
    print("starting traing loop")
    for i in range(epochs):
        loss = torch.tensor(0.0).to(device)

        j=0
        for d in loader:
            d = d.to(device)
            L = laplacian(d).to(device)
            x = f(d).to(device)
            loss += loss_f(x,L).to(device)/update
            j+=1

            if j%update.item()==0 or j==len(loader):
                optimizer.zero_grad()
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
                loss = torch.tensor(0.0).to(device)
        
        if i%print_loss_every==0:
            print("Epoch:",i,"Loss:",losses[-1])
    print("Training complete")

    #save the model
    #torch.save(f.state_dict(), os.path.join(modelpath,"spectralnet.pth"))
    torch.save(f.state_dict(), modelpath)
    print("Model saved as spectralnet.pth")


