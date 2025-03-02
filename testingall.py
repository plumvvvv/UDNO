#from phase1training import trainingloop as trainingloop1
import scipy.sparse
import scipy.sparse.linalg
from phase1net import SpectralNet
from phase2net import PermNet
from scipy.io import mmread
import torch
import os
import networkx as nx
import torch_geometric
import torch_geometric.utils
from torch_geometric.utils import get_laplacian
import numpy as np
import scipy
import random
import time
#import torch._dynamo
#torch._dynamo.config.suppress_errors = True

torch.manual_seed(176364)
np.random.seed(453658)
random.seed(41884)

datapath = "/home/ml4a/code"#code\\dl-spectral-graph-partitioning\\"#/dl-spectral-graph-partitioning"
modelpath = "/home/ml4a/code/dl-spectral-graph-partitioning/softrankmodels"
device =  'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'

spectralnet_path = os.path.join(modelpath,"spectralnet.pth")
f = SpectralNet().to(device)

f.load_state_dict(torch.load(spectralnet_path))
f.eval()
for p in f.parameters():
    p.requires_grad = False
f.eval()

#compiled_f = torch.compile(f,dynamic=True)


sepnet_path = os.path.join(modelpath,"permnetlr0.0001s0.0001b1d0.0000001.minexactlossate46.pth")
f_perm = PermNet().to(device)
f_perm.load_state_dict(torch.load(sepnet_path))
f_perm.eval()
for p in f_perm.parameters():
    p.requires_grad = False
f_perm.eval()

#compiled_f_perm = torch.compile(f_perm,dynamic=True)

print("Model loaded")

def laplacian(graph):
    lap = get_laplacian(graph.edge_index,
                        num_nodes=graph.num_nodes,
                        normalization='rw')#left normalized laplacian
    a=lap[0].numpy(force=True)
    b=lap[1].numpy(force=True)

    return scipy.sparse.csc_matrix((b,(a[0],a[1])))

#testing suitsparse in Mats2024
#suitsparsepath = os.path.join(datapath,"Mats2024extract")
#suitsparsepath = os.path.join(datapath,"Mats2024extract1+2")
suitsparsepath = os.path.join(datapath,"Mats2024extract3")
for matrixname in os.listdir(suitsparsepath):
    print(matrixname)
    matrixfile = matrixname+".mtx"
    fullmatrixfile = os.path.join(suitsparsepath,matrixname,matrixfile)
    matrix = mmread(fullmatrixfile)
    _,_,_,_,_,symm = scipy.io.mminfo(fullmatrixfile)
    if symm!='symmetric' :
        if  matrix.shape[0]==matrix.shape[1]:
            matrix=scipy.sparse.coo_matrix((1/2)*(matrix+matrix.transpose()))
        else:
            continue
    nxg = nx.from_scipy_sparse_array(matrix)
    print("Graph "+matrixname)
    #print(type(matrix))
    if nx.is_connected(nxg):
        lutimes = []
        ordertimes = []
        nnzs=[]
        nnzs_n=[]
        #initialize edges
        edges = torch.tensor(np.array([matrix.row,matrix.col]),dtype=torch.long)
        edges,_ = torch_geometric.utils.remove_self_loops(edges)
        #initialize node feature vectors with random values
        nodes = torch.randn(matrix.shape[0],2)
        #initialize graph data batch for gnn
        graph = torch_geometric.data.Batch(x=nodes,edge_index=edges).to(device)
        g = torch_geometric.utils.to_networkx(graph,to_undirected=True)
        graph.batch = torch.zeros(graph.num_nodes,dtype=torch.long).to(device)

        
        #print("test graph done")
        a=nx.to_scipy_sparse_array(g,format='csc')#+scipy.sparse.identity(g.number_of_nodes())
        #a=matrix.tocsc()
        #raw
        try:
            #luresult = scipy.sparse.linalg.splu(, permc_spec='NATURAL')
            t0 = time.time()
            luresult=scipy.sparse.linalg.splu(a, permc_spec="NATURAL")
            t1 = time.time()
            lutime = t1-t0
            ordertime = -1
            nnz = (luresult.L.nnz + luresult.U.nnz - a.nnz)*1.0/a.nnz
            nnz_n = (luresult.L.nnz + luresult.U.nnz - a.nnz)*1.0/a.shape[0]/a.shape[1]
            nnzs.append(nnz)
            nnzs_n.append(nnz_n)
            lutimes.append(lutime)
            ordertimes.append(ordertime)
            del luresult
        except:
            nnzs.append(-1)
            nnzs_n.append(-1)
            lutimes.append(-1)
            ordertimes.append(-1)
            continue
            
        print("Natural lutime:",lutimes[-1])

        #test amdbar
        import ctypes
        libamd = ctypes.cdll.LoadLibrary('amd/build/libAMDWrapper.so')
        b = matrix.tocsr()
        num_nodes = b.shape[0]
        b_indptr = b.indptr.astype(np.int32)
        b_indices = b.indices.astype(np.int32)
        perm = np.zeros(num_nodes, dtype=np.int32)
        iperm = np.zeros(num_nodes, dtype=np.int32)
        t0 = time.time()
        libamd.WRAPPER_amd(
            ctypes.c_int(num_nodes),
            b_indptr.ctypes.data_as(ctypes.c_void_p),
            b_indices.ctypes.data_as(ctypes.c_void_p),
            perm.ctypes.data_as(ctypes.c_void_p),
            iperm.ctypes.data_as(ctypes.c_void_p)
        )
        perm = perm.tolist()
        t1=time.time()
        ordertime=t1-t0
        permeda = scipy.sparse.csc_matrix(a[:, perm][perm, :], dtype=np.int32)
        try:
            t0=time.time()
            luresult = scipy.sparse.linalg.splu(permeda,permc_spec='NATURAL')
            t1=time.time()
            lutime = t1-t0
            nnz = (luresult.L.nnz + luresult.U.nnz - a.nnz)*1.0/a.nnz
            nnz_n = (luresult.L.nnz + luresult.U.nnz - a.nnz)*1.0/a.shape[0]/a.shape[1]
            nnzs.append(nnz)
            nnzs_n.append(nnz_n)
            lutimes.append(lutime)
            ordertimes.append(ordertime)
            del luresult
        except:
            nnzs.append(-1)
            nnzs_n.append(-1)
            lutimes.append(-1)
            ordertimes.append(ordertime) 
        del perm
        del permeda
    




        #test metis
        import nxmetis
        t0 = time.time() 
        perm = nxmetis.node_nested_dissection(g)
        t1 = time.time()
        ordertime = t1 - t0
        permeda = scipy.sparse.csc_matrix(a[:, perm][perm, :], dtype=np.int32)
        try:
            t0=time.time()
            luresult = scipy.sparse.linalg.splu(permeda, permc_spec='NATURAL')
            t1 = time.time()
            lutime = t1-t0
            nnz = (luresult.L.nnz + luresult.U.nnz - a.nnz)*1.0/a.nnz
            nnz_n = (luresult.L.nnz + luresult.U.nnz - a.nnz)*1.0/a.shape[0]/a.shape[1]
            nnzs.append(nnz)
            nnzs_n.append(nnz_n)
            lutimes.append(lutime)
            ordertimes.append(ordertime)
            del luresult
        except:
            nnzs.append(-1)
            nnzs_n.append(-1)
            lutimes.append(-1)
            ordertimes.append(ordertime)
        del perm
        del permeda



        #exact fiedler score
        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html
        t0 = time.time()
        Lgraph = laplacian(graph)
        lamb,V = scipy.sparse.linalg.eigs(Lgraph,k=2,which='SM')
        fiedlerscore = V[:,1]
        t1 = time.time()
        ordertime = t1 - t0
        pairs=[(i, s) for i,s in enumerate(fiedlerscore) ]
        pairs.sort(key=lambda it:it[1])
        perm = [p[0] for p in pairs]
        permeda = scipy.sparse.csc_matrix(a[:, perm][perm, :], dtype=np.int32)	
        try:
            t0 = time.time()
            luresult = scipy.sparse.linalg.splu(permeda, permc_spec='NATURAL')
            t1 = time.time()
            lutime = t1-t0
            nnz = (luresult.L.nnz + luresult.U.nnz - a.nnz)*1.0/a.nnz
            nnz_n = (luresult.L.nnz + luresult.U.nnz - a.nnz)*1.0/a.shape[0]/a.shape[1]
            nnzs.append(nnz)
            nnzs_n.append(nnz_n)
            lutimes.append(lutime)
            ordertimes.append(ordertime)
            del luresult
        except:
            nnzs.append(-1)
            nnzs_n.append(-1)
            lutimes.append(-1)
            ordertimes.append(ordertime)
        del perm
        del permeda


        
        t0 = time.time()
        graph.x = f(graph)[:,1].reshape((graph.num_nodes,1))#fielder vector
        #graph.x = compiled_f(graph)[:,1].reshape((graph.num_nodes,1))#fielder vector
        graph.x = (graph.x - torch.mean(graph.x))*torch.sqrt(torch.tensor(graph.num_nodes))
        predicted_fielder_score = graph.x
        t1 = time.time()
        ordertime = t1 - t0
        pairs=[(i, s) for i,s in enumerate(predicted_fielder_score) ]
        pairs.sort(key=lambda it:it[1])
        perm = [p[0] for p in pairs]
        permeda = scipy.sparse.csc_matrix(a[:, perm][perm, :], dtype=np.int32)	
        try:
            t0 = time.time()
            luresult = scipy.sparse.linalg.splu(permeda, permc_spec='NATURAL')
            t1 = time.time()
            lutime = t1-t0
            nnz = (luresult.L.nnz + luresult.U.nnz - a.nnz)*1.0/a.nnz
            nnz_n = (luresult.L.nnz + luresult.U.nnz - a.nnz)*1.0/a.shape[0]/a.shape[1]
            nnzs.append(nnz)
            nnzs_n.append(nnz_n)
            lutimes.append(lutime)
            ordertimes.append(ordertime)
            del luresult
        except:
            nnzs.append(-1)
            nnzs_n.append(-1)
            lutimes.append(-1)
            ordertimes.append(ordertime)
        del perm
        del permeda
        #graph.x = compiled_f(graph)[:,1].reshape((graph.num_nodes,1))#fielder vector

        t0 = time.time()
        score = f_perm(graph)
        #score = compiled_f_perm(graph)
        t1 = time.time()
        ordertime = ordertimes[-1] + t1 - t0
        #score = compiled_f_perm(graph)
        pairs = [(i, s) for i,s in enumerate(score) ]
        pairs.sort(key=lambda it:it[1], reverse=True)
        perm = [p[0] for p in pairs]
        permeda = scipy.sparse.csc_matrix(a[:, perm][perm, :], dtype=np.int32)	
        try:
            t0 = time.time()
            luresult = scipy.sparse.linalg.splu(permeda, permc_spec='NATURAL')
            t1 = time.time()
            lutime = t1-t0
            nnz = (luresult.L.nnz + luresult.U.nnz - a.nnz)*1.0/a.nnz
            nnz_n = (luresult.L.nnz + luresult.U.nnz - a.nnz)*1.0/a.shape[0]/a.shape[1]
            nnzs.append(nnz)
            nnzs_n.append(nnz_n)
            lutimes.append(lutime)
            ordertimes.append(ordertime)
            del luresult
        except:
            nnzs.append(-1)
            nnzs_n.append(-1)
            lutimes.append(-1)
            ordertimes.append(ordertime)
        del score
        del pairs
        del permeda
        del perm



        

            
        print(nnzs)
        print(nnzs_n)
        print(lutimes)
        print(ordertimes)








