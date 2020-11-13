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
        print(int(ret_torch[0][i]), end="\t")
    print("\t")
    for i in range(20):
        print(int(ret_torch[1][i]), end="\t")
    print("\t")
    return ret_torch

def load_tsv(tsv_path: str, video_id_set):
    #video_id -> account
    video_id2acc = dict()
    account2info = dict()
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
            cat_1 = fields[3]
            if video_id_str not in video_id_set:
                line = fp.readline()
                continue

            if video_id_str not in video_id2acc:
                video_id2acc[video_id_str] = []
            video_id2acc[video_id_str].append(account_str)
            account2info[account_str] = cat_1
            line = fp.readline()
#    for key in ret_dict.keys():
#        print("{}\t{}".format(key, len(ret_dict[key])))

    return video_id2acc, account2info

def make_graph(tsv_path:str, feature_path: str):
    file_list = os.listdir(feature_path)


    video_name_list_full = [cur_str.split(".")[0] for cur_str in file_list]
    video_name_list_set = set(video_name_list_full)
    video_id2acc, _ = load_tsv(tsv_path, video_name_list_set)



    video_name_list = []
    for video_id in video_name_list_full:
        if video_id in video_id2acc:
            video_name_list.append(video_id)#to make sure that video_name_list is the intersection of two parts.


    print("A total of {} videos found.".format(len(video_name_list)))
    idx2video_name = dict()
    video_name2idx = dict()

    video_idx = 0
    video_cnt = 0
    for i in range(len(video_name_list)):
        cur_video_id = video_name_list[i]
        for j in range(FRAME_CNT):
            idx2video_name[video_idx + j] = cur_video_id
        video_cnt += 1
        video_idx += FRAME_CNT

    for key in idx2video_name.keys():
        cur_video_id = idx2video_name[key]
        if cur_video_id not in video_name2idx:
            video_name2idx[cur_video_id] = []
        video_name2idx[cur_video_id].append(key)#key is the index here


    
    ret_pic_node = makePicPair(video_cnt, FRAME_CNT)
    g = dgl.heterograph({('pic', 'nb', 'pic'): (ret_pic_node[0], ret_pic_node[1])})
    #g = dgl.heterograph({('pic', 'nb', 'pic'):(torch.tensor([0, 0, 2]), torch.tensor([1, 2, 3])), ('acc', 'own', 'pic'):(torch.tensor([0, 1, 0]), torch.tensor([1, 2, 3]))})
    pic_node_num = g.num_nodes('pic')
    print(pic_node_num)
    #acc_node_num = g.num_nodes('acc')
    #print("Current number of node {} and {}.".format(pic_node_num, acc_node_num))
    #g.nodes['pic'].data['h'] = torch.ones(pic_node_num, 1)
    #g.nodes['acc'].data['h'] = torch.ones(acc_node_num, 1)
    #print(g.successors(0, etype='own'))
    #print(g.ndata['h'][0])
    #print(g.ndata['h'][3])
    #print(g.ndata['h'][10])
    #print(g.num_nodes())


if __name__=="__main__":
    make_graph(sys.argv[1], sys.argv[2])
