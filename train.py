import os
import numpy as np
import cv2
import torch
import copy
import sys
import pkbar
import random
import dgl
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from model import GNNRankModel
from pytorchtools import EarlyStopping
from dgl.data.utils import load_graphs
import dgl.function as fn

def make_train_step(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(l, r, y):
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        y_pred = model(l, r)
        # Computes loss
        loss = loss_fn(y_pred, y)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item(), y_pred

    # Returns the function that will be called inside the train loop
    return train_step

def calculate_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    return acc

def train(root_dir: str, meta_data_path: str, batch_size: int):
    if not torch.cuda.is_available():
        print("Not available.")
        exit()
    epoch_cnt = 10
    torch.cuda.init()
    device_id = torch.cuda.current_device()
    print("Device: {}, ID: {}, Avalability: {}".format(torch.cuda.get_device_name(device_id), str(device_id), torch.cuda.is_available()))
    glist, label_dict = load_graphs("./test_data.bin")
    print(type(glist), type(label_dict))
    print(len(glist))
    g = glist[0]
    print(g.etypes)

    pic_feats = g.nodes['pic'].data['img_feat']
    acc_feats = g.nodes['acc'].data['acc_feat']
    edge_num = g.num_edges('nb')
    train_edge_num = edge_num * 0.8
    train_eid_dict = {'nb': torch.arange(0, int(train_edge_num))}
    print(type(pic_feats), type(acc_feats))
    print(pic_feats.size())
    print(acc_feats.size())

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_eid_dict, sampler,
        #exclude='reverse_types',
        #reverse_etypes={'pb': 'blt'},
        batch_size=64,
        shuffle=True,
        drop_last=False,
        num_workers=4)
    """
    idx = 0
    for input_nodes, edge_subgraph, blocks in dataloader:
        for key in input_nodes.keys():
            print(key, input_nodes[key].size())
        print(len(input_nodes), len(blocks), idx)
        idx += 1
    """
    in_features = 1000
    hidden_features = 512
    out_features = 128
    num_classes = 1
    model = GNNRankModel(in_features, hidden_features, out_features, num_classes, g.etypes)
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(epoch_cnt):
        print("Epoch {}".format(epoch))
        for input_nodes, edge_subgraph, blocks in dataloader:
            blocks = [b.to(torch.device('cuda')) for b in blocks]
            edge_subgraph = edge_subgraph.to(torch.device('cuda'))

            pic_feats = blocks[0].nodes['pic'].data['img_feat']
            acc_feats = blocks[0].nodes['acc'].data['acc_feat']

            #input_features = blocks[0].srcdata['features']
            edge_num = edge_subgraph.num_edges('nb')
            edge_labels = edge_subgraph.edata['label']
            #print(type(pic_feats), type(acc_feats))
            print(pic_feats.size(), acc_feats.size(), edge_num, type(edge_labels))
            #edge_predictions = model(edge_subgraph, blocks, input_features)
            #loss = compute_loss(edge_labels, edge_predictions)
            #opt.zero_grad()
            #loss.backward()
            #opt.step()
    exit()


    model = ReOrderingModel(1000, 512, 64, g.etypes)


    label = g.edges['nb'].data['label']
    train_mask = g.edges['nb'].data['train_mask']
    node_features = {'pic': pic_feats, 'acc': acc_feats}

    opt = torch.optim.Adam(model.parameters())
    for epoch in range(10):
        pred = model(g, node_features, 'nb')
        loss = ((pred[train_mask] - label[train_mask]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())

    exit()



    train_dataset = MakeDataset(root_dir, meta_data_list[0: train_len])
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = MakeDataset(root_dir, meta_data_list[train_len: ])
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    epoch = 100
    batch_size = batch_size

    rank_model = RankNet(1)
    rank_model = rank_model.cuda()
    torch.backends.cudnn.benchmark = False
    rank_model.eval()

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(rank_model.parameters())
    train_step = make_train_step(rank_model, loss_fn, optimizer)

    early_stopping = EarlyStopping(patience=5, verbose=True)

    train_per_epoch = (train_len // batch_size)

    for i in range(epoch):
        """
        print("Epoch: {}".format(str(i)))
        rank_model.zero_grad()
        diff_pred = rank_model(clip_l, clip_r)
        loss = loss_fun(diff_pred, diff)
        loss.backward()
        optimizer.step()
        """
        kbar = pkbar.Kbar(target=train_per_epoch, epoch=i, num_epochs=epoch, width=8, always_stateful=False)
        idx = 0
        loss_list = []
        acc_list = []
        for clip_l_batch, clip_r_batch, diff_batch in train_data_loader:

            # the dataset "lives" in the CPU, so do our mini-batches
            # therefore, we need to send those mini-batches to the
            # device where the model "lives"
            if torch.cuda.is_available():
                clip_l_batch = clip_l_batch.cuda()
                clip_r_batch = clip_r_batch.cuda()
                diff_batch = diff_batch.cuda()
                # print("Batch: {}".format(str(idx)))
                loss, y_pred = train_step(clip_l_batch, clip_r_batch, diff_batch)
                acc = calculate_acc(y_pred, diff_batch)
                loss_list.append(loss)
                acc_list.append(acc)

                idx += 1
                kbar.update(idx, values=[("loss", loss), ("accuracy", acc)])
        avg_train_loss = sum(loss_list) / len(loss_list)
        avg_train_acc = sum(acc_list) / len(acc_list)

        idx = 0
        acc_list = []
        for clip_l_batch, clip_r_batch, diff_batch in val_data_loader:

            # the dataset "lives" in the CPU, so do our mini-batches
            # therefore, we need to send those mini-batches to the
            # device where the model "lives"
            if torch.cuda.is_available():
                clip_l_batch = clip_l_batch.cuda()
                clip_r_batch = clip_r_batch.cuda()
                diff_batch = diff_batch.cuda()
                # print("Batch: {}".format(str(idx)))
                y_pred = rank_model(clip_l_batch, clip_r_batch)
                acc = calculate_acc(y_pred, diff_batch)
                acc_list.append(acc)

                idx += 1
        avg_val_acc = sum(acc_list) / len(acc_list)

        kbar.add(1, values=[("train_accuracy", avg_train_acc), ("val_accuracy", avg_val_acc)])
        #print("Epoch: {}, train loss: {}, train accuracy: {}, val accuracy: {}".format(str(i), str(avg_train_loss), str(avg_train_acc), str(avg_val_acc)))
        early_stopping(avg_val_acc, rank_model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    torch.save(rank_model.state_dict(), "./test_saving_model.pt")










    #train_data, val_data = torch.utils.data.random_split(full_data, (train_len, valid_len), generator=torch.Generator().manual_seed(42))






"""
    with torch.no_grad():
        result_pred = rank_model.predict(clip_r)
        print(result_pred)
        print(result_pred.size())



    train_data_l = torch.load(pt_path)
"""
def transform(snippet):
    ''' stack & noralization '''
    snippet = np.concatenate(snippet, axis=-1)
    snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
    snippet = snippet.mul_(2.).sub_(255).div(255)

    return snippet.view(1,-1,3,snippet.size(1),snippet.size(2)).permute(0,2,1,3,4)


if __name__ == '__main__':
    train(sys.argv[1],sys.argv[2], int(sys.argv[3]))
