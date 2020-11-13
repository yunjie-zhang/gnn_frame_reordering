import numpy as np
import random
import torch
import dgl
import sys

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

def make_graph(feature_path: str):
    g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
    g.add_nodes(2)
    print(g.num_nodes())


if __name__=="__main__":
    video2acc_dict = load_tsv(sys.argv[1])
    make_graph("abc")
