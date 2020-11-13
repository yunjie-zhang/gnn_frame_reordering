import numpy as np
import random
import torch
import dgl
import sys
import os

FRAME_CNT = 8

def makePicPair(total_cnt, interval):
    base_pair_list = []
    pair_list = []
    for i in range(interval):
        for j in range(interval):
            if i != j:
                base_pair_list.append([i, j])
    for i in range(total_cnt):
        base_idx = i * interval
        for base_pair in base_pair_list:
            pair_list.append([base_pair[0] + base_idx, base_pair[1] + base_idx])
    print("A total of {} pairs.".format(len(pair_list)))
    ret_torch = torch.tensor(pair_list, dtype=torch.int)
    ret_torch = torch.transpose(ret_torch, 0, 1)
    print("Size of torch tensor is {}".format(ret_torch.shape))
    for i in range(20):
        print(ret_torch[0][i], end="\t")
    print("\t")
    for i in range(20):
        print(ret_torch[1][i], end="\t")

    return ret_torch

def load_tsv(tsv_path: str):
    #video_id -> account
    ret_dict = dict()
    with open(tsv_path, "r") as fp:
        line = fp.readline()
        line = fp.readline()
        while line:
            fields = line.split("\t")
            if len(fields) < 2:
                line = fp.readline()
                continue

            account_str = fields[0]
            video_id_str = fields[1]

            if video_id_str not in ret_dict:
                ret_dict[video_id_str] = []
            ret_dict[video_id_str].append(account_str)

            line = fp.readline()
#    for key in ret_dict.keys():
#        print("{}\t{}".format(key, len(ret_dict[key])))

    return ret_dict

def make_graph(feature_path: str, video2acc_dict):
    file_list = os.listdir(feature_path)
    print("A total of {} videos found.".format(len(file_list)))

    video_name_list = [cur_str.split(".")[0] for cur_str in file_list]
    idx2video_name = dict()

    video_idx = 0
    video_cnt = 0
    for i in range(len(video_name_list)):
        cur_video_id = video_name_list[i]
        for j in range(FRAME_CNT):
            idx2video_name[video_idx + j] = cur_video_id
        video_cnt += 1
        video_idx += FRAME_CNT

    makePicPair(video_cnt, FRAME_CNT)

    g = dgl.heterograph({('pic', 'nb', 'pic'):(torch.tensor([0, 0, 2]), torch.tensor([1, 2, 3])), ('acc', 'own', 'pic'):(torch.tensor([0, 1, 0]), torch.tensor([1, 2, 3]))})
    pic_node_num = g.num_nodes('pic')
    acc_node_num = g.num_nodes('acc')
    print("Current number of node {} and {}.".format(pic_node_num, acc_node_num))
    g.nodes['pic'].data['h'] = torch.ones(pic_node_num, 1)
    g.nodes['acc'].data['h'] = torch.ones(acc_node_num, 1)
    print(g.successors(0, etype='own'))
    #print(g.ndata['h'][0])
    #print(g.ndata['h'][3])
    #print(g.ndata['h'][10])
    print(g.num_nodes())


if __name__=="__main__":
    video2acc_dict = load_tsv(sys.argv[1])
    make_graph(sys.argv[2], video2acc_dict)
