import torch
from torch import nn
from torch.nn import Parameter
from torch.autograd import Variable
from torch.nn.functional import binary_cross_entropy

from poincare_distance import poincare_distance


class BiasedEmbedding(nn.Module):
    def __init__(self, n_feat, n_dim):
        self.vect = nn.Embedding(n_feat, n_dim)
        self.bias = nn.Embedding(n_feat, 1)

    def __call__(self, index):
        return self.bias(index), self.vect(index)


class PFM(nn.Module):
    def __init__(self, n_user, n_item, n_dim):
        self.embed_user = nn.BiasedEmbedding(n_user, n_dim)
        self.embed_item = nn.BiasedEmbedding(n_item, n_dim)
        self.bias = Parameter(Variable(torch.ones(1)))

    def forward(self, u, i, r):
        bu, vu = self.embed_user(u)
        bi, vi = self.embed_user(i)
        dist = poincare_distance(vu, vi)
        logodds = self.bias + bi + bu + dist
        loss = binary_cross_entropy(logodds, r)
        return loss
