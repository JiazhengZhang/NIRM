from numpy.core.fromnumeric import shape
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from Model import *
import torch
import numpy as np
from Utils import *
from SyntheticDataset import *
from torch_geometric.data import DataLoader
from scipy import stats
import argparse
import random

class TrainDataset:
    def __init__(self):

        self.TRAIN_DATA_PATH = os.path.join(os.getcwd(), 'data', 'train', 'dataset')

    def ReadTrainFile(self):

        feature = os.path.join(self.TRAIN_DATA_PATH, 'train_dataset_feature.npy')
        adj = os.path.join(self.TRAIN_DATA_PATH, 'train_dataset_adj.npy')
        label = os.path.join(self.TRAIN_DATA_PATH, 'train_dataset_label.npy')
        num_graph = len(np.array(pickle_read(feature), dtype=object))
        return feature, adj, label, num_graph

    def CreateDataset(self):

        feature, adj, label, num_graph = self.ReadTrainFile()
        syn_dataset = SyntheticDataset(root='./' + 'SYN_Dataset', Adj=adj,
                                       node_feature=feature, graph_label=label)
        train_dataset = syn_dataset[:round(num_graph * 0.9)]
        test_dataset = syn_dataset[round(num_graph * 0.9):]


        return train_dataset, test_dataset



def train():
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        data = data.to(device)
        out = model(data.x, data.edge_index, data.num_nodes)
        loss = criterion(out, data.y.view(-1, 1))
        # print(out.shape)
        # print(data.y.view(-1,1).shape)
        # print(data.x.shape)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients


def kendall_rank_coffecient(out, label, batch_size, data):
    sum = 0
    last_split = 0
    for i in range(batch_size):
        batch_node = data[i].num_nodes
        out_node = out[last_split: last_split + batch_node]
        label_node = label[last_split: last_split + batch_node]
        out_rank = np.argsort(out_node, axis=0).reshape(-1)
        label_rank = np.argsort(label_node, axis=0).reshape(-1)
        tau, p_value = stats.kendalltau(label_rank, out_rank)
        sum += tau
        last_split = last_split + batch_node

    return sum


def test(loader):
    model.eval()
    loss = 0
    rank = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        out = model(data.x, data.edge_index, data.num_nodes)
        loss += criterion(out, data.y.view(-1, 1))
        rank += kendall_rank_coffecient(out.cpu().detach().numpy(), data.y.view(-1, 1).cpu().detach().numpy(),
                                        data.num_graphs, data)

    return loss / len(loader.dataset), rank / len(loader.dataset)


def save_model(epoch):
    checkpoint = {"model_state_dict": model.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
                  "epoch": epoch}
    path_checkpoint = os.path.join(CHECKPOINTS_PATH, "checkpoint_{}_epoch.pkl".format(epoch))
    torch.save(checkpoint, path_checkpoint)






if __name__ == '__main__':


    model_name = 'NIRM'
    loss_type = 'MSE'
    graph_type = 'MIXED'
    num_node_features = 5





    TrainSets = TrainDataset()
    train_dataset, test_dataset = TrainSets.CreateDataset()
    train_loader = DataLoader(train_dataset, batch_size= 5, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size= 1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NIRM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = torch.nn.MSELoss(reduction='mean')

    scheduler_1 = StepLR(optimizer, step_size=50, gamma=0.3)
    epoch_num = 200
    checkpoint_interval = 5
    CHECKPOINTS_PATH = os.path.join(os.getcwd(), 'training', model_name)
    os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
    LOGGING_PATH = CHECKPOINTS_PATH + '/Logs/'
    os.makedirs(LOGGING_PATH, exist_ok=True)
    NIRM_logger = get_logger(LOGGING_PATH + 'train.log')
    os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
    patience_period = 10
    BEST_VAL_LOSS = 100
    BEST_TRA_LOSS = 100
    BEST_VAL_KEN = 0
    BEST_TRA_KEN = 0
    BEST_TRA_MODEL_EPOCH = 0
    BEST_VAL_MODEL_EPOCH = 0

    PATIENCE_CNT = 0

    for epoch in range(epoch_num):
        train()
        loss1 = test(train_loader)
        loss2 = test(test_loader)

        # scheduler_1.step(loss2[0])
        # scheduler_1.step()
        if loss2[0] < BEST_VAL_LOSS or loss1[0] < BEST_TRA_LOSS or loss2[1] > BEST_VAL_KEN or loss1[1] > BEST_TRA_KEN:
            if loss2[0] < BEST_VAL_LOSS:
                BEST_VAL_MODEL_EPOCH = epoch
            elif loss1[0] < BEST_TRA_LOSS :
                BEST_TRA_MODEL_EPOCH = epoch
            elif loss2[1] > BEST_VAL_KEN:
                NIRM_logger.info(
                    'Better_VAL_Kendall_EPOCH:[{}/{}] \t Kendal:[{}/{}]'.format(
                        epoch, epoch_num, loss2[1], BEST_VAL_KEN))
                BEST_VAL_KEN= loss2[1]
            elif loss1[1] > BEST_TRA_KEN:
                NIRM_logger.info(
                    'Better_TAR_Kendall_EPOCH:[{}/{}] \t Kendal:[{}/{}]'.format(
                        epoch, epoch_num, loss1[1], BEST_TRA_KEN))
                BEST_TRA_KEN = loss1[1]

            BEST_VAL_LOSS = min(loss2[0], BEST_VAL_LOSS)  # keep track of the best validation accuracy so far
            BEST_TRA_LOSS = min(loss1[0], BEST_TRA_LOSS)
            PATIENCE_CNT = 0  # reset the counter every time we encounter new best accuracy
            save_model(epoch)
        else:
            PATIENCE_CNT += 1  # otherwise keep counting

        if PATIENCE_CNT >= patience_period:
            NIRM_logger.info(
                'BEST_TRA_MODEL_EPOCH:[{}/{}]\t BEST_VAL_MODEL_EPOCH:[{}/{}]\t TestLoss={:.6f}'.format(BEST_TRA_MODEL_EPOCH, epoch_num, BEST_VAL_MODEL_EPOCH, epoch_num,
                                                                                                       BEST_VAL_LOSS))
            raise Exception('Stopping the training, the universe has no more patience for this training.')



        NIRM_logger.info(
            'Epoch:[{}/{}]\t TrainLoss={:.6f}\t TrainKendal={:.4f}\t TestLoss={:.6f}\t TestKendal={:.4f}'.format(epoch,
                                                                                                                 epoch_num,
                                                                                                                 loss1[0],
                                                                                                                 loss1[1],
                                                                                                                 loss2[0],
                                                                                                                 loss2[1]))
    NIRM_logger.info('finish training!')