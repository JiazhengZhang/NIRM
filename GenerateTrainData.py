import argparse
import multiprocessing as mp
import networkx as nx
from itertools import combinations
import numpy as np
from igraph import *
np.seterr(divide='ignore',invalid='ignore')
import collections
from Utils import *



def ExhaustiveSearch(G, nodes_number):

    original_largest_cc = len(max(nx.connected_components(G), key=len))
    basic_element = {i for i in range(nodes_number)}

    optimal_sets = []
    number_flag = nodes_number
    exit_flag = False
    for number_attack in range(nodes_number):

        all_possible = list(list(i) for i in combinations(basic_element, number_attack + 1))
        print( 'Cost:', len(all_possible[0]))
        for item in all_possible:
            G_copy = nx.Graph(G)
            G_copy.remove_nodes_from(item)
            residual_largest_cc = len(max(nx.connected_components(G_copy), key=len))

            if residual_largest_cc <= round(original_largest_cc * 0.20):
                print('We have achieve the goal !')
                print(item)
                number_flag = len(item)
                optimal_sets.append(item)
                exit_flag = True

            else:
                if len(item) > number_flag:
                    break
        if exit_flag == True:


            return optimal_sets


def Counts_High_Order_Nodes(G, depth = 2):
    NODES_LIST = list(G.nodes)
    output = {}
    output = output.fromkeys(NODES_LIST)
    for node in NODES_LIST:
        layers = dict(nx.bfs_successors(G, source=node, depth_limit=depth))
        high_order_nodes = sum([len(i) for i in layers.values()])
        output[node] = high_order_nodes

    return output


def Generate_Node_Feature(G):
    NODES_LIST = list(G.nodes)

    # the number of its one-hop neighbors
    degree_dict = nx.degree_centrality(G)
    degree_list = np.array([degree_dict[i] for i in NODES_LIST])[:, None]
    degree_list = degree_list / np.max(degree_list)

    # the number of its two-hop neighbors
    second_neighbor = Counts_High_Order_Nodes(G, depth=2)
    second_neighbor_list = np.array([second_neighbor[i] for i in NODES_LIST])[:, None]
    second_neighbor_list = second_neighbor_list/np.max(second_neighbor_list)

    # average degree of its one-hop neighbors
    neighbor_average_degree = nx.average_neighbor_degree(G)
    neighbor_average_degree_list = np.array([neighbor_average_degree[i] for i in NODES_LIST])[:, None]
    neighbor_average_degree_list = neighbor_average_degree_list / np.max(neighbor_average_degree_list)

    # local clustering coefficient
    local_clustering_dict = nx.clustering(G)
    local_clustering_list = np.array([local_clustering_dict[i] for i in NODES_LIST])[:, None]

    # constant
    constant_list = np.ones((len(NODES_LIST), 1))

    node_features = np.concatenate((degree_list, second_neighbor_list, neighbor_average_degree_list, local_clustering_list,
                                    constant_list), axis=1)

    node_features[np.isnan(node_features)] = 0

    return node_features


def Generate_Graph(g_type, num_min = 20, num_max = 26):
    num_nodes = np.random.randint(num_max - num_min + 1) + num_min

    if g_type == 'erdos_renyi':
        g = nx.erdos_renyi_graph( n=num_nodes, p=0.1)
    elif g_type == 'small-world':
        g = nx.connected_watts_strogatz_graph(n=num_nodes, k=4, p=0.1)
    elif g_type == 'barabasi_albert':
        g = nx.barabasi_albert_graph(n=num_nodes, m=3)
    elif g_type == 'powerlaw':
        g = nx.powerlaw_cluster_graph(n=num_nodes, m=3, p=0.05)

    g.remove_nodes_from(list(nx.isolates(g)))
    g.remove_edges_from(nx.selfloop_edges(g))
    num_nodes = len(g.nodes)

    return g, num_nodes


def Label_Rank(score):
    score = score.reshape(-1)
    rank = np.argsort(score)
    norm_label = np.empty(rank.shape)
    for j in range(len(rank)):
        norm_label[rank[j]] = float(j+1) / float(len(rank))
    norm_label = np.array(norm_label).reshape(-1)

    return norm_label


def Label_Normalied_Rank(score, g):

    degree_dict = {n: d for n, d in g.degree()}
    Nodes = list(g.nodes())
    score = score.reshape(-1)
    out = np.sort(score)
    scorecount = collections.Counter(out)


    rank = np.argsort(score)
    norm_label = np.empty(rank.shape)
    i = 0
    while i < len(rank) :
        if scorecount[out[i]] == 1 and score[rank[i]] == out[i]:
            norm_label[rank[i]] = float(i+1 ) / float(len(rank))
            i = i + 1

        elif scorecount[out[i]] != 1:
            SameScoreNodes = [Nodes[rank[j]] for j in range(i, i + scorecount[out[i]])]
            SameNodesDegree = np.array([degree_dict[s] for s in SameScoreNodes])
            DegreeRank = np.argsort(SameNodesDegree)
            for j in range(len(DegreeRank)):
                norm_label[Nodes.index(SameScoreNodes[DegreeRank[j]])] = float(i + 1 ) / float(len(rank))
                i = i + 1

    norm_label = np.array(norm_label).reshape(-1)

    return norm_label




def Training_Score_Propagation(optimal_sets, g, g_adj):

    # Initial Training Score
    nodes = list(g.nodes)
    num_nodes = len(nodes)
    initial_score = [0] * num_nodes
    for optimal_set in optimal_sets:
        for item in optimal_set:
            initial_score[nodes.index(item)] += 1
    initial_score = np.array(initial_score).reshape(-1,1)
    initial_score = initial_score / np.max(initial_score)


    # Training Score Propagation
    degree = g_adj.sum(axis=0, keepdims=True)
    propagtion_score = np.matmul(g_adj / degree , initial_score).reshape(-1)
    training_score = initial_score + propagtion_score

    return training_score




def Save_OptimalSets(OptimalSets, DATA_DIR_PATH, id):
    OptimalFile = os.path.join(DATA_DIR_PATH, 'OptimalSetsFile')
    os.makedirs(OptimalFile, exist_ok=True)
    File = os.path.join(OptimalFile, 'train_optimal_sets_'+str(id)+'.npy')

    pickle_save( File, OptimalSets)


def Save_Graph_GML(G, DATA_DIR_PATH, id):
    GraphFile = os.path.join(DATA_DIR_PATH, 'SourceGraph')
    os.makedirs(GraphFile, exist_ok=True)
    File = os.path.join(GraphFile, str(id)+'.gml')
    nx.write_gml(G, File)



def GenerateTrainData(id, graph_type):

    print(f'Generating No.{id} training {graph_type} graphs')

    DATASET_PATH = os.path.join(os.getcwd(), 'data', 'train', '20_25_'+graph_type+'_graph')
    os.makedirs(DATASET_PATH, exist_ok=True)

    if graph_type == 'ER':
        g_type = 'erdos_renyi'
    elif graph_type == 'WS':
        g_type = 'small-world'
    elif graph_type == 'BA':
        g_type = 'barabasi_albert'
    elif graph_type == 'PLC':
        g_type = 'powerlaw'

    # Generate Graph
    g, num_nodes = Generate_Graph(g_type = g_type)
    Save_Graph_GML(g, DATASET_PATH, id)

    # Generate Adj, Features
    g_adjacent_matrix = np.array(nx.adjacency_matrix(g).todense())
    g_features = Generate_Node_Feature(g)

    # Exhaustive search for the optimal solutions
    optimal_sets = ExhaustiveSearch(g, num_nodes)
    Save_OptimalSets(optimal_sets, DATASET_PATH, id)
    print('-------------------------------------------------------------------------------')
    print(f'No.{id} Graph has searched its optimal soulutions: {len(optimal_sets)} sets')

    # Generate Label
    score = Training_Score_Propagation(optimal_sets, g, g_adjacent_matrix)
    label = Label_Normalied_Rank(score, g)

    pickle_save(os.path.join(DATASET_PATH, 'train_adj_'+str(id)+'.npy'), g_adjacent_matrix)
    pickle_save(os.path.join(DATASET_PATH, 'train_feature_' + str(id) + '.npy'), g_features)
    pickle_save(os.path.join(DATASET_PATH, 'train_label_' + str(id) + '.npy'),   label)


def PrepareTrainData(graph_type_list, num_graph):

    Adj = []
    Label =  []
    Feature = []

    for graph_type in graph_type_list:
        DATA_PATH = os.path.join(os.getcwd(), 'data', 'train', '20_25_' + graph_type + '_graph')
        for id in range(num_graph):
            File_Path = os.path.join(DATA_PATH, 'train_adj_'+str(id)+'.npy')
            if os.path.isfile(File_Path):
                adj_id = pickle_read(os.path.join(DATA_PATH,'train_adj_'+str(id)+'.npy'))
                feature_id = pickle_read(os.path.join(DATA_PATH, 'train_feature_'+str(id)+'.npy'))
                label_id = pickle_read(os.path.join(DATA_PATH, 'train_label_'+str(id)+'.npy'))
                Adj.append(adj_id)
                Feature.append(feature_id)
                Label.append(label_id[:, None])

    SAVE_PATH = os.path.join(os.getcwd(), 'data', 'train', 'dataset')
    os.makedirs(SAVE_PATH, exist_ok=True)
    pickle_save(os.path.join(SAVE_PATH, 'train_dataset_adj.npy'), Adj)
    pickle_save(os.path.join(SAVE_PATH, 'train_dataset_label.npy'), Label)
    pickle_save(os.path.join(SAVE_PATH, 'train_dataset_feature.npy'), Feature)




if __name__ == '__main__':

    Synthetic_Type = ['BA', 'ER', 'PLC', 'WS']
    num_graph = 1000
    for type in Synthetic_Type:
        for id in range(num_graph):
            GenerateTrainData(id, type)

    PrepareTrainData(Synthetic_Type, num_graph)
