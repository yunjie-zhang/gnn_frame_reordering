import numpy as np
import random
import torch
import dgl
import sys
import os
from torch import nn
import torch.nn.functional as F
import dgl.nn as dglnn
import dgl.function as fn

class StochasticTwoLayerRGCN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, rel_names):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_feat, hidden_feat, norm='both',
                                      weight=True, bias=True, activation=nn.ReLU())
        self.conv2 = dglnn.GraphConv(hidden_feat, out_feat, norm='both',
                                      weight=True, bias=True, activation=nn.ReLU())

    def forward(self, blocks, x):
        x = self.conv1(blocks[0], x)
        x = self.conv2(blocks[1], x)
        return x

class ScorePredictor(nn.Module):
    def __init__(self, num_classes, in_features):
        super().__init__()
        self.W = nn.Linear(in_features * 2, 1)
        self.S = nn.Sigmoid()

    def apply_edges(self, edges):
        data = torch.cat([edges.src['x'], edges.dst['x']], 1)
        #print(data.size())
        return {'score': torch.squeeze(self.S(self.W(data)))}

    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            #for etype in edge_subgraph.canonical_etypes:
            edge_subgraph.apply_edges(self.apply_edges, etype='nb')
            return edge_subgraph.edata['score']
        
class DotProductPredictor(nn.Module):
    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']
        
class GNNRankModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_classes, rel_names):
        super().__init__()
        self.gcn = StochasticTwoLayerRGCN(
            in_features, hidden_features, out_features, rel_names)
        self.predictor = DotProductPredictor()

    def forward(self, edge_subgraph, blocks, x):
        x = self.gcn(blocks, x)
        return self.predictor(edge_subgraph, x)

#model = GNNRankModel(in_features, hidden_features, out_features, num_classes)
