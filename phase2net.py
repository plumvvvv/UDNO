'''
Partitioning Module
'''
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data,Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv,graclus,avg_pool



class PermNet(torch.nn.Module):
    def __init__(self):
        super(PermNet, self).__init__()

        self.l = 16
        self.pre = 2
        self.post = 2
        self.coarsening_threshold = 2
        self.activation = torch.tanh
        self.lins=[16,16,16,16,16]

        #self.pre gnn layers
        self.conv_first = SAGEConv(1,self.l)#why 1? only for fielder vector
        self.conv_pre = nn.ModuleList([SAGEConv(self.l,self.l) for i in range(self.pre)])

        #self.post gnn layers
        self.conv_post = nn.ModuleList([SAGEConv(self.l,self.l) for i in range(self.post)])
        self.conv_coarse = SAGEConv(self.l,self.l)

        self.lins1 = nn.Linear(self.l,self.lins[0])
        self.lins2 = nn.Linear(self.lins[0],self.lins[1])
        self.lins3 = nn.Linear(self.lins[1],self.lins[2])
        self.final = nn.Linear(self.lins[4],1)##why self.lins[4]?
        #output ranking scores.

    def forward(self,graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        x = self.activation(self.conv_first(x,edge_index))
        x_info = []
        cluster_info = []
        edge_info = []
        batches = []
        while x.size()[0] > self.coarsening_threshold:
            #pre-smoothing: self.pre gnn layers
            for i in range(self.pre):
                x = self.activation(self.conv_pre[i](x,edge_index))
            
            #pooling/coarsening/restriction
            x_info.append(x)
            batches.append(batch)
            cluster = graclus(edge_index,num_nodes=x.shape[0])
            cluster_info.append(cluster)
            edge_info.append(edge_index)
            gc = avg_pool(cluster, Batch(batch=batch,x=x,edge_index=edge_index))
            x,edge_index,batch = gc.x,gc.edge_index,gc.batch
        #coarse iterations
        x = self.activation(self.conv_coarse(x,edge_index))
        while edge_info:
            #un-pooling/interpolation/prolongation/refinement
            edge_index = edge_info.pop()
            output,inverse = torch.unique(cluster_info.pop(),return_inverse=True)
            x = (x[inverse]+x_info.pop())/2
            #post-smoothing: self.post gnn layers
            for i in range(self.post):
                x = self.activation(self.conv_post[i](x,edge_index))
        
        x = self.activation(self.lins1(x))
        x = self.activation(self.lins2(x))
        x = self.activation(self.lins3(x))
        x = self.final(x)
        #x = torch.sigmoid(x)
        return x