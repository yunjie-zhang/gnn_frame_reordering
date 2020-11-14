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

def loadTsv(tsv_path: str, video_name_list_full):
    #video_id -> account
    video_name_list_set = set(video_name_list_full)
    video_from_tsv = set()

    account2video = dict()
    account2info = dict()
    with open(tsv_path, "r") as fp:
        line = fp.readline()#the first line is title
        line = fp.readline()
        while line:
            fields = line.split("\t")
            if len(fields) < 2:
                line = fp.readline()
                continue

            account_str = fields[0]
            video_id_str = fields[1]
            cat_1 = fields[3]
            if video_id_str not in video_name_list_set:
                line = fp.readline()
                continue

            if account_str not in account2video:
                account2video[account_str] = []
            account2video[account_str].append(video_id_str)
            account2info[account_str] = cat_1
            video_from_tsv.add(video_id_str)
            line = fp.readline()
#    for key in ret_dict.keys():
#        print("{}\t{}".format(key, len(ret_dict[key])))

    video_name_list = []
    video_candidate_set = video_from_tsv & video_name_list_set

    for video_id in video_candidate_set:
        video_name_list.append(video_id)


    return account2video, account2info, video_name_list

def makeAccPair(account2video, video_name2idx):
    account2idx = dict()
    acc_idx = 0
    for acc_str in account2video.keys():
        account2idx[acc_str] = acc_idx
        acc_idx += 1

    account2video_list = []
    for acc in account2video.keys():
        for video_id in account2video[acc]:
            account2video_list.append([acc, video_id])

    account2pic_list = []
    for acc_video_pair in account2video_list:
        acc_id = acc_video_pair[0]
        video_id = acc_video_pair[1]
        for pic_id in video_name2idx[video_id]:
            account2pic_list.append([acc_id, pic_id])

    acc_idx2pic_list = []
    pic2acc_idx_list = []
    for pair in account2pic_list:
        acc_idx = account2idx[pair[0]]
        pic_idx = pair[1]
        acc_idx2pic_list.append([acc_idx, pic_idx])
        pic2acc_idx_list.append([pic_idx, acc_idx])

    ret_torch_1 = torch.tensor(acc_idx2pic_list, dtype=torch.int)
    ret_torch_1 = torch.transpose(ret_torch_1, 0, 1)
    print("Size of torch tensor is {}".format(ret_torch_1.shape))

    ret_torch_2 = torch.tensor(pic2acc_idx_list, dtype=torch.int)
    ret_torch_2 = torch.transpose(ret_torch_2, 0, 1)
    print("Size of torch tensor is {}".format(ret_torch_2.shape))

    return ret_torch_1, ret_torch_2, account2idx






def make_graph(tsv_path:str, feature_path: str):
    file_list = os.listdir(feature_path)
    video_name_list_full = [cur_str.split(".")[0] for cur_str in file_list]





    account2video, account2info, video_name_list = loadTsv(tsv_path, video_name_list_full)

    print("A total of {} accounts found.".format(len(account2video)))
    print("A total of {} accounts found.".format(len(account2info)))
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

    for cur_video_id in video_name2idx:
        if len(video_name2idx[cur_video_id]) != FRAME_CNT:
            print("Frame count doesn't match.")



    ret_acc2pic_node, ret_pic2acc_node, account2idx = makeAccPair(account2video, video_name2idx)
    ret_pic_node = makePicPair(video_cnt, FRAME_CNT)

    g = dgl.heterograph({('pic', 'nb', 'pic'): (ret_pic_node[0], ret_pic_node[1]),
                         ('acc', 'pb', 'pic'): (ret_acc2pic_node[0], ret_acc2pic_node[1]),
                         ('pic', 'blt', 'acc'): (ret_pic2acc_node[0], ret_pic2acc_node[1])})
    #g = dgl.heterograph({('pic', 'nb', 'pic'):(torch.tensor([0, 0, 2]), torch.tensor([1, 2, 3])), ('acc', 'own', 'pic'):(torch.tensor([0, 1, 0]), torch.tensor([1, 2, 3]))})
    pic_node_num = g.num_nodes('pic')
    acc_node_num = g.num_nodes('acc')
    print("Total pictures count {}".format(pic_node_num))
    print("Total accounts count {}".format(acc_node_num))
    g.nodes['pic'].data['img_feat'] = torch.ones(pic_node_num, 1024)
    g.nodes['acc'].data['acc_feat'] = torch.ones(acc_node_num, 1024)

    save_graphs("./test_data.bin", g)
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
