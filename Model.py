from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn as nn
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul

# Embedding Dimension 1-32-16-8
# Attention Head      8-4-2
class NIRM(torch.nn.Module):
    def __init__(self):
        super(NIRM, self).__init__()

        self.conv1 = GATConv( 1, 4, heads=8, concat=True, negative_slope=0.2, dropout=0.1)
        self.conv2 = GATConv( 32, 4, heads=4, concat=True, negative_slope=0.2, dropout=0.1)
        self.conv3 = GATConv( 16, 4, heads=2, concat=True, negative_slope=0.2, dropout=0.2)

        self.lin1 = Linear(5, 1, bias=True)
        self.lin2 = Linear(8, 1, bias=False)
        self.activation = nn.ELU()

    def forward(self, x, edge_index, num_nodes):
        fill_value = 1
        Adj_matrix = SparseTensor(row=edge_index[0], col=edge_index[1],
                                  sparse_sizes=(num_nodes, num_nodes))
        Adj_matrix = fill_diag(Adj_matrix, fill_value)
        Adj = F.normalize(Adj_matrix.to_dense(), p=1, dim=1)

        # Feature Scoring
        init_socre = self.lin1(x)

        # Encoding Representation
        x2 = self.conv1(init_socre, edge_index)
        x3 = self.conv2(x2, edge_index)
        x4 = self.conv3(x3, edge_index)
        x5 = F.dropout(x4, p=0.3, training=self.training)

        # Local Scoring
        R = torch.matmul(x5, x5.t())
        R_1 = torch.mul(R, Adj)
        R_2 = torch.sum(R_1, dim=1, keepdim=True)
        normalied_degree = x[:, 0].view(-1, 1)
        local_score = R_2 + normalied_degree

        # Global Socring
        R_3 = torch.matmul(Adj, x5)
        global_score = self.lin2(R_3)

        ranking_scores = torch.add(local_score, global_score)
        return ranking_scores

