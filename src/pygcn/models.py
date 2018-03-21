import torch
import torch.nn as nn
import torch.nn.functional as F
from src.pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, embeddings):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout
        self.embeddings = torch.autograd.Variable(torch.FloatTensor(embeddings))

    def forward(self, adj):
        x = F.relu(self.gc1(self.embeddings, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
        # return F.log_softmax(x)
