from phase2net import PermNet
from trainingdataset import mixed_dataset
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data,Batch
from torch_geometric.loader import DataLoader
from phase2loss import loss_normalized_cut
import os
import numpy as np
import networkx as nx
import scipy

#device = 'GPU' if torch.cuda.is_available() else 'CPU'
def constructtrainingdataset(f,datapath,device):
    ng = 20
    n_iter,n_iter_suitesparse = 5,1
    listdata,count_fem,count_suitesparse = mixed_dataset(ng,200,500,n_iter,n_iter_suitesparse,datapath)
    trainingdataset = []
    for adj in listdata:
        row = adj.row
        col = adj.col
        rowcols = np.array([row,col])
        edges = torch.tensor(rowcols,dtype=torch.long)
        edges,_ = torch_geometric.utils.remove_self_loops(edges)
        #initialize node feature vectors with random values
        nodes = torch.randn(adj.shape[0],2)
        graph = Batch(x=nodes,edge_index=edges).to(device)
        graph.x = f(graph)[:,1].reshape((graph.num_nodes,1))#fielder vector
        graph.x = (graph.x - torch.mean(graph.x))*torch.sqrt(torch.tensor(graph.num_nodes))
        trainingdataset.append(graph)
    return trainingdataset,ng,count_fem,count_suitesparse




def trainingloopForPermNetwithRankloss(f,phase2loss_fn,datapath,modelpath,device):
    print("Starting Partitioning Module Training ")
    f_part = PermNet().to(device)
    print("Number of parameters:", sum(p.numel() for p in f_part.parameters()))

    #loss function
    loss_f_perm = phase2loss_fn

    #training dataset
    #construct training dataset
    trainingdataset,ng,count_fem,count_suitesparse = constructtrainingdataset(f,datapath,device)
    loader = DataLoader(trainingdataset,batch_size=1,shuffle=True,pin_memory=False)
    print("Training dataset for partitioning module done")
    print("Number of training graphs:", len(loader))
    print("Number of Delanauy graphs:", ng+int(ng/2))
    print("Number of FEM graphs:", count_fem)
    print("Number of SuiteSparse graphs:", count_suitesparse)

    natural_exactnnz =0
    natural_exactnnz_n = 0
    natural_numnnz=0
    for graph in trainingdataset:
        graph = graph.to(device)
        g = torch_geometric.utils.to_networkx(graph,to_undirected=True)
        a=nx.to_scipy_sparse_array(g,format='csc')#+10*scipy.sparse.identity(g.number_of_nodes())
        try:
            luresult = scipy.sparse.linalg.splu(a, permc_spec='NATURAL')
            natural_exactnnz += (luresult.L.nnz + luresult.U.nnz - a.nnz)*1.0/a.nnz
            natural_exactnnz_n += (luresult.L.nnz + luresult.U.nnz - a.nnz)*1.0/a.shape[0]/a.shape[1]
            natural_numnnz+=1
        except:
            continue
    print("natural nnz:",natural_exactnnz/natural_numnnz,"natural nnz_n:",\
          natural_exactnnz_n/natural_numnnz)
    
    
    init_exactnnz =0
    init_exactnnz_n = 0
    init_numnnz=0
    for graph in trainingdataset:
        score = graph.x
        g = torch_geometric.utils.to_networkx(graph,to_undirected=True)
        a=nx.to_scipy_sparse_array(g,format='csc')
        pairs = [(i, s) for i,s in enumerate(score)]
        pairs.sort(key=lambda it:it[1], reverse=True)
        perm = [p[0] for p in pairs]
        permeda = scipy.sparse.csc_matrix(a[:, perm][perm, :], dtype=np.int32)
        try:
            luresult = scipy.sparse.linalg.splu(permeda, permc_spec='NATURAL')
            init_exactnnz += (luresult.L.nnz + luresult.U.nnz - a.nnz)*1.0/a.nnz
            init_exactnnz_n += (luresult.L.nnz + luresult.U.nnz - a.nnz)*1.0/a.shape[0]/a.shape[1]
            init_numnnz+=1
        except:
            continue
    print("init nnz:",init_exactnnz/init_numnnz,"init nnz_n:",init_exactnnz_n/init_numnnz)

    initr_exactnnz =0
    initr_exactnnz_n = 0
    initr_numnnz=0
    for graph in trainingdataset:
        g = torch_geometric.utils.to_networkx(graph,to_undirected=True)
        a=nx.to_scipy_sparse_array(g,format='csc')#+10*scipy.sparse.identity(g.number_of_nodes())
        score = graph.x
        pairs = [(i, s) for i,s in enumerate(score)]
        pairs.sort(key=lambda it:it[1])
        perm = [p[0] for p in pairs]
        permeda = scipy.sparse.csc_matrix(a[:, perm][perm, :], dtype=np.int32)
        try:
            luresult = scipy.sparse.linalg.splu(permeda, permc_spec='NATURAL')
            initr_exactnnz += (luresult.L.nnz + luresult.U.nnz - a.nnz)*1.0/a.nnz
            initr_exactnnz_n += (luresult.L.nnz + luresult.U.nnz - a.nnz)*1.0/a.shape[0]/a.shape[1]
            initr_numnnz+=1
        except:
            continue
    print("initr nnz:",initr_exactnnz/initr_numnnz,"initr nnz_n:",initr_exactnnz_n/initr_numnnz)

    #optimizer
    init_lr = 0.0001
    optimizer = torch.optim.Adam(f_part.parameters(), lr=init_lr,weight_decay=0.001)
    epochs = 100#100 #500
    losses = []
    exactlosses = []
    update = torch.tensor(1).to(device) #steps after which the loss function is updated
    print_loss_every = 1 #50 steps after which the loss is printed.

    #training loop
    print("starting traing partition module loop")
    print("init_lr="+str(init_lr))
    f_part.train()
    for i in range(epochs):
        loss = torch.tensor(0.0).to(device)
        exactloss = torch.tensor(0.0)
        j=0
        #print("lr = "+str(optimizer.param_groups[0]['lr']))
        for d in loader:
            d = d.to(device)
            y_pred = f_part(d)
            loss += loss_f_perm(y_pred,d)/update

            pairs=[(i, s) for i,s in enumerate(y_pred) ]
            pairs.sort(key=lambda it:it[1], reverse=True)
            perm = [p[0] for p in pairs]
            y_rank = torch.zeros(len(perm),dtype=torch.long)
            y_rank[perm] = torch.tensor(range(len(perm)))
            exactloss += torch.sum(torch.abs(y_rank[d.edge_index[0]]-y_rank[d.edge_index[1]]))/update
            
            j+=1

            if j%update.item()==0 or j==len(loader):
                optimizer.zero_grad()
                losses.append(loss.item())
                exactlosses.append(exactloss.item())
                loss.backward()
                optimizer.step()
                loss = torch.tensor(0.0).to(device)
                exactloss = torch.tensor(0.0)
        
        if i%print_loss_every==0:
            
            exactnnz =0
            exactnnz_n = 0
            numnnz=0
            igraph=0
            for graph in trainingdataset:
                graph = graph.to(device)
                y_pred = f_part(graph)
                if(igraph==0):
                    print(y_pred[0:5].squeeze())
                g = torch_geometric.utils.to_networkx(graph,to_undirected=True)
                a=nx.to_scipy_sparse_array(g,format='csc')#+10*scipy.sparse.identity(g.number_of_nodes())
                pairs=[(i, s) for i,s in enumerate(y_pred) ]
                pairs.sort(key=lambda it:it[1], reverse=True)
                perm = [p[0] for p in pairs]
                permeda = scipy.sparse.csc_matrix(a[:, perm][perm, :], dtype=np.int32)
                try:
                    luresult = scipy.sparse.linalg.splu(permeda, permc_spec='NATURAL')
                    exactnnz += (luresult.L.nnz + luresult.U.nnz - a.nnz)*1.0/a.nnz
                    exactnnz_n += (luresult.L.nnz + luresult.U.nnz - a.nnz)*1.0/a.shape[0]/a.shape[1]
                    numnnz+=1
                except:
                    continue
                igraph += 1
            print("Epoch:",i,"Loss:",losses[-1],"Exact Loss:",exactlosses[-1],\
               "exact nnz:",exactnnz/numnnz," exact nnz_n:",exactnnz_n/numnnz)
#        if torch.abs(torch.tensor(losses[-1]-losses[-2]))<0.000001:
#            break
    print("Training partition module complete")

    #save the model
    #torch.save(f_part.state_dict(), os.path.join(modelpath,"permnet.pth"))
    torch.save(f_part.state_dict(), modelpath)
    print("Model saved as permnet.pth")

