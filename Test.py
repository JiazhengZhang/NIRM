
import torch
from Model import *
import math
from Model import *
from Utils import *
import os
import networkx as nx
import numpy as np
from torch_geometric.data import Data

class Dismatnle:
    def __init__(self, model, NetWorkList, TEST_DATA_PATH, TEST_RESULT_PATH):


        self.RESULT_PATH = TEST_RESULT_PATH
        os.makedirs(self.RESULT_PATH, exist_ok=True)
        self.Test_PATH = TEST_DATA_PATH
        self.Network_list = NetWorkList

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model.to(self.device)

    def CreateGraph(self, NETWORK_PATH, filename):
        G = nx.Graph()
        for line in open(os.path.join(NETWORK_PATH, filename + '.txt')):
            strlist = line.split()
            n1 = int(strlist[0])
            n2 = int(strlist[1])
            G.add_edges_from([(n1, n2)])
        return G





    def RegenGraphData(self, G):
        graph = nx.create_empty_copy(G)
        graph.add_edges_from(list(G.edges))
        Adj = np.array(nx.adjacency_matrix(G).todense())
        source_nodes, target_nodes = np.nonzero(Adj[:, :])
        source_nodes = source_nodes.reshape((1, -1))
        target_nodes = target_nodes.reshape((1, -1))
        edge_index = torch.tensor(np.concatenate((source_nodes, target_nodes), axis=0), dtype=torch.long)
        node_feature = self.generate_node_feature(G)
        node_feature[np.isnan(node_feature)] = 0
        x = torch.tensor(node_feature[:, :], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index.view(2, -1), num_nodes=G.number_of_nodes())

        return data

    def counts_high_order_nodes(self, G, depth=2):
        NODES_LIST = list(G.nodes)
        output = {}
        output = output.fromkeys(NODES_LIST)
        for node in NODES_LIST:
            layers = dict(nx.bfs_successors(G, source=node, depth_limit=depth))
            high_order_nodes = 0
            for i in layers.keys():
                high_order_nodes += len(layers[i])
            # high_order_nodes = sum([len(i) for i in layers.values()])
            output[node] = high_order_nodes

        return output

    def generate_node_feature(self, G):

        NODES_LIST = list(G.nodes)

        degree_dict = nx.degree_centrality(G)
        degree_list = np.array([degree_dict[i] for i in NODES_LIST])[:, None]
        degree_list = degree_list / np.max(degree_list)


        second_neighbor = self.counts_high_order_nodes(G, depth=2)
        second_neighbor_list = np.array([second_neighbor[i] for i in NODES_LIST])[:, None]
        second_neighbor_list = second_neighbor_list / np.max(second_neighbor_list)


        neighbor_average_degree = nx.average_neighbor_degree(G)
        neighbor_average_degree_list = np.array([neighbor_average_degree[i] for i in NODES_LIST])[:, None]
        neighbor_average_degree_list = neighbor_average_degree_list / np.max(neighbor_average_degree_list)

        local_clustering_dict = nx.clustering(G)
        local_clustering_list = np.array([local_clustering_dict[i] for i in NODES_LIST])[:, None]


        constant_list = np.ones((len(NODES_LIST), 1))

        node_features = np.concatenate(
            (degree_list, second_neighbor_list, neighbor_average_degree_list, local_clustering_list,
             constant_list), axis=1)

        node_features[np.isnan(node_features)] = 0

        return node_features


    def onepass_dismantle(self):

        for Network in self.Network_list:
            os.makedirs(self.RESULT_PATH, exist_ok=True)

            # NETWORK_PATH = os.path.join(self.Test_PATH, Network)
            NETWORK_PATH = self.Test_PATH
            G = self.CreateGraph(NETWORK_PATH, Network)
            original_largest_cc = G.number_of_nodes()
            Graph_data = self.RegenGraphData(G)
            Graph_data.to(self.device)
            nodes = list(G.nodes)

            out = self.model(Graph_data.x, Graph_data.edge_index, Graph_data.num_nodes)
            scores = out.view(-1)
            num_nodes = scores.nelement()
            pred_values, pred_indices = scores.topk(num_nodes)
            target_index = pred_indices.cpu().numpy()
            target_node_index = np.array([nodes[i] for i in list(target_index)])

            TAS = []
            TAS_CON = []

            for node in target_node_index:

                G.remove_node(node)
                if len(G) == 0:
                    residual_largest_cc = 0
                else:
                    residual_largest_cc = len(max(nx.connected_components(G), key=len)) / original_largest_cc
                TAS_CON.append(residual_largest_cc)
                TAS.append(node)
            self.WriteTAS(os.path.join(self.RESULT_PATH, Network + ".txt"), Network, TAS, TAS_CON, True)

    def WriteTAS(self, File, Network, TAS, TAS_CON, flag):

        with open(File, 'w') as f:
            for key in range(len(TAS)):
                if flag:
                    f.writelines(str(TAS[key]) + ' ' + str(TAS_CON[key]))
                    f.write('\n')
            f.close()




def LoadModel(path, model):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(path, map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


if __name__=="__main__":

    NetWorkList = ['AirTraffic', 'Bible', 'Infectious']

    path = os.path.join(os.getcwd(), 'checkpoints', 'NIRM_onepass.pkl')
    dismantle_strategy = 'onepass'
    model = NIRM()
    MODEL = LoadModel(path, model)
    TEST_DATA_PATH = os.path.join(os.getcwd(), 'data', 'realworld')
    TEST_RESULT_PATH = os.path.join(os.getcwd(), 'result', 'realworld', dismantle_strategy)

    Test = Dismatnle(MODEL, NetWorkList, TEST_DATA_PATH, TEST_RESULT_PATH)
    Test.onepass_dismantle()

