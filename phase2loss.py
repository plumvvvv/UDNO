import torch
import torch_geometric
from torch_geometric.utils import degree
from torch.distributions import Normal

def loss_normalized_cut(y_pred,graph):

    d = degree(graph.edge_index[0],num_nodes=y_pred.size(0))
    gamma = y_pred.t() @ d
    c = torch.sum(y_pred[graph.edge_index[0],0] * y_pred[graph.edge_index[1],1])
    return torch.sum(torch.div(c,gamma))


def loss_vertex_separator(y_pred,graph):
    #y_pred[:,0],y_pred[:,1],y_pred[:,2]
    E_A = torch.sum(y_pred[:,0] )
    E_B = torch.sum(y_pred[:,1] )
    E_S = torch.sum(y_pred[:,2] )
    return torch.sum(torch.div(E_S,E_A)+torch.div(E_S,E_B)) 



    
def loss_softrank(y_pred,graph):
    y = y_pred[:,0].squeeze()
    num_nodes = y_pred.shape[0]
    sigma = torch.sqrt(torch.tensor(2.0))*torch.tensor(0.0001)
    ub_0 = torch.tensor(0.0)#,requires_grad=True)
    Er = torch.cat(
        [
            (
                torch.sum(
                    1.0-Normal(y[i]-y,sigma).cdf(ub_0)
                )-0.5
            ).reshape(1)
            for i in range(num_nodes)
        ],
        dim=0
    )
    Dr = torch.cat(
        [
            (
                torch.sum(
                    Normal(y[i]-y,sigma).cdf(ub_0)*(1.0 - Normal(y[i]-y,sigma).cdf(ub_0))
                ) - 0.5*0.5
            ).reshape(1)
            for i in range(num_nodes)
        ],
        dim = 0
    )
    E_rx_minus_ry = Er[graph.edge_index[0]]-Er[graph.edge_index[1]]
    #D_rx_minus_ry = Dr[graph.edge_index[0]]+Dr[graph.edge_index[1]]
    D_rx_minus_ry = torch.clamp(Dr[graph.edge_index[0]]+Dr[graph.edge_index[1]],min=1e-38)
    nd_rx_minus_ry = Normal(E_rx_minus_ry,torch.sqrt(D_rx_minus_ry))
    return torch.sum(
        2*D_rx_minus_ry*nd_rx_minus_ry.log_prob(ub_0).exp() + 
        E_rx_minus_ry*(1.0-2*nd_rx_minus_ry.cdf(ub_0))
    )/num_nodes
