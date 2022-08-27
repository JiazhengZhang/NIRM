from Utils import pickle_read
from Utils import  pickle_save
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data


class SyntheticDataset(InMemoryDataset):

    def __init__(self, root, Adj, node_feature, graph_label, transform=None, pre_transform=None):
        self.Adj = np.array(pickle_read(Adj), dtype=object)
        self.node_feature = np.array(pickle_read(node_feature), dtype=object)
        self.graph_label_lpa_r = np.array(pickle_read(graph_label), dtype=object)
        self.num_graph = len(self.graph_label_lpa_r)
        super(SyntheticDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [r'.\Synthetic.dataset']

    def download(self):
        pass

    def process(self):
        data_list = []  # graph classification need to define data_list for multiple graph
        for i in range(self.num_graph):
            source_nodes, target_nodes = np.nonzero(self.Adj[i])
            source_nodes = source_nodes.reshape((1, -1))
            target_nodes = target_nodes.reshape((1, -1))

            edge_index = torch.tensor(np.concatenate((source_nodes, target_nodes), axis=0),
                                      dtype=torch.long)  # edge_index should be long type

            feature_x = self.node_feature[i]
            feature_x[np.isnan(feature_x)] = 0
            x = torch.tensor(feature_x, dtype=torch.float)


            label_y = self.graph_label_lpa_r[i]
            label_y[np.isnan(label_y)] = 0
            y = torch.tensor(label_y, dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])