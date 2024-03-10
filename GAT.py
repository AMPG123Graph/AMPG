import torch
import torch.nn as nn
import torch.nn.functional as F
from graphlayers import GraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.1, alpha=0.2, nheads=1, concat=True):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.concat = concat
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=concat) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        if self.concat:
            x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        else:
            x = torch.stack([att(x, adj) for att in self.attentions], dim=0).mean(dim=0)
        x = F.dropout(x, self.dropout, training=self.training)
        return x