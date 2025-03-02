import numpy as np
import torch
import networkx as nx
import scipy.io as sio
import os
import scipy
import scipy.spatial 
import itertools
import random

# Seeds
torch.manual_seed(176364)
np.random.seed(453658)
random.seed(41884)


'''
create the dataset made of Delaunay squares and rectangles, FEM triangulations and SuiteSparse matrices
role:
    generate the Delaunay squares and rectangles
    load the FEM and SuiteSparse matrices from the directory datapath
Input: 
      n/2 - the number of delaunay rectangles in the dataset
         n - the number of delaunay squares in the dataset
     n_min - the minimum number of nodes in the dataset
     n_max - the maximum number of nodes in the dataset
     n_repeat_fem - the number of times for each matrix from the FEM directory
     n_repeat_suitesparse - the number of times for each matrix from the SuiteSparse directory
    datapath - the path to the directory containing the FEM and SuiteSparse matrices
Output: 
     listdataset - the list of matrices in the dataset
        count_fem - the number of FEM matrices in the dataset
    count_suitesparse - the number of SuiteSparse matrices in the dataset

For unsymmetric matrices:
    matrix = mmread(os.path.join(suitsparsepath,matrixfile))
    _,_,_,_,_,symm = scipy.io.mminfo(os.path.join(suitsparsepath,matrixfile))
    if symm!='symmetric':
        matrix=scipy.sparse.coo_matrix((1/2)*(matrix+matrix.transpose()))
added by 20240909
'''
def mixed_dataset(n,n_min,n_max,n_repeat_fem,n_repeat_suitesparse,datapath):
    listdataset = []
    count_fem,count_suitesparse = 0,0
    #Generate Delaunay rectangles
    for i in range(int(n/2)):
        num_nodes = np.random.choice(np.arange(n_min,n_max+1,2))
        points = np.random.random_sample((num_nodes,2))
        twos = np.full((num_nodes,1),2)
        ones = np.ones((num_nodes,1))
        resize = np.concatenate((twos,ones),axis=1)
        points = resize*points
        g = graph_delaunay_from_points(points)
        listdataset.append(nx.to_scipy_sparse_array(g,format='coo',dtype=float))
    #Generate Delaunay squares
    for i in range(n):
        num_nodes = np.random.choice(np.arange(n_min,n_max+1,2))
        points = np.random.random_sample((num_nodes,2))
        g = graph_delaunay_from_points(points)
        listdataset.append(nx.to_scipy_sparse_array(g,format='coo',dtype=float))
    #load FEM trigualations dataset from directory
    fem_path_graded_l = os.path.expanduser(os.path.join(datapath,'graded_l'))
    for m in os.listdir(fem_path_graded_l):
        adj = sio.mmread(os.path.join(fem_path_graded_l,str(m)))
        _,_,_,_,_,symm = scipy.io.mminfo(os.path.join(fem_path_graded_l,str(m)))
        if symm!='symmetric':
            adj=scipy.sparse.coo_matrix((1/2)*(adj+adj.transpose()))
        if adj.shape[0]<n_max and adj.shape[0]>n_min:
            for i in range(n_repeat_fem):
                listdataset.append(adj)
                count_fem += 1
    fem_path_hole3 = os.path.expanduser(os.path.join(datapath,'hole3'))
    for m in os.listdir(fem_path_hole3):
        adj = sio.mmread(os.path.join(fem_path_hole3,str(m)))
        _,_,_,_,_,symm = scipy.io.mminfo(os.path.join(fem_path_hole3,str(m)))
        if symm!='symmetric':
            adj=scipy.sparse.coo_matrix((1/2)*(adj+adj.transpose()))
        if adj.shape[0]<n_max and adj.shape[0]>n_min:
            for i in range(n_repeat_fem):
                listdataset.append(adj)
                count_fem += 1
    fem_path_hole6 = os.path.expanduser(os.path.join(datapath,'hole6'))
    for m in os.listdir(fem_path_hole6):
        adj = sio.mmread(os.path.join(fem_path_hole6,str(m)))
        _,_,_,_,_,symm = scipy.io.mminfo(os.path.join(fem_path_hole6,str(m)))
        if symm!='symmetric':
            adj=scipy.sparse.coo_matrix((1/2)*(adj+adj.transpose()))
        if adj.shape[0]<n_max and adj.shape[0]>n_min:
            for i in range(n_repeat_fem):
                listdataset.append(adj)
                count_fem += 1
    suitsparse_path = os.path.expanduser(os.path.join(datapath,'suitsparse'))
    for m in os.listdir(suitsparse_path):
        adj = sio.mmread(os.path.join(suitsparse_path,str(m)))
        _,_,_,_,_,symm = scipy.io.mminfo(os.path.join(suitsparse_path,str(m)))
        if symm!='symmetric':
            adj=scipy.sparse.coo_matrix((1/2)*(adj+adj.transpose()))
        g = nx.from_scipy_sparse_array(adj)
        if adj.shape[0]<n_max and adj.shape[0]>n_min and nx.is_connected(g):
            for i in range(n_repeat_suitesparse):
                listdataset.append(adj)
                count_suitesparse += 1
    #load SuiteSparse matrices dataset from directory
    print("Number of FEM matrices:",count_fem)
    print("Number of SuiteSparse matrices:",count_suitesparse)
    return listdataset,count_fem,count_suitesparse


'''
role:
     generate the Delaunay graph from the points.
Input: 
    points - the points of the graph
Output: 
         g - the networkx graph of the Delaunay mesh
'''
def graph_delaunay_from_points(points):
    mesh = scipy.spatial.Delaunay(points,qhull_options="QJ")
    mesh_simp = mesh.simplices
    edges = []
    for i in range(len(mesh_simp)):
        edges += itertools.combinations(mesh_simp[i],2)
    e = list(set(edges))
    g = nx.Graph(e)
    return g