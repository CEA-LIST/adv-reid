# @copyright CEA-LIST/DIASI/SIALV/LVA (2020)
# @author CEA-LIST/DIASI/SIALV/LVA <quentin.bouniot@cea.fr>
# @license CECILL

import numpy as np
import torch
from torch import nn


def get_distances(dist, ids):
    """
    Compute the largest positive distance and smallest negative distance for 
    each element as anchor and returns the batch of positive distance and 
    negative distance.
    Args:
        dist (Tensor): Matrix of the pairwise distances of the batch.
        ids (list): List of the ids of each batch instance.
    Returns:
        batch of largest positive distance
        batch of smallest negative distance
        list of triplets
    """
    dist_an = []
    dist_ap = []
    list_triplet = []
    for index_i, id_i in enumerate(ids):
        max_pos_dist = 0
        pos_pair = (0,0)
        min_neg_dist = np.inf
        neg_pair = (0,0)
        for index_j, id_j in enumerate(ids):
            if index_j == index_i:
                continue
            if id_i == id_j:
                if dist[index_i][index_j] > max_pos_dist:
                    max_pos_dist = dist[index_i][index_j]
                    pos_pair = (index_i, index_j)
            else:
                if dist[index_i][index_j] < min_neg_dist:
                    min_neg_dist = dist[index_i][index_j]
                    neg_pair = (index_i, index_j)
        dist_ap.append(dist[pos_pair[0]][pos_pair[1]])
        dist_an.append(dist[neg_pair[0]][neg_pair[1]])
        list_triplet.append((pos_pair[0],pos_pair[1],neg_pair[1]))
    dist_ap = torch.stack(dist_ap)
    dist_an = torch.stack(dist_an)
    return dist_ap, dist_an, list_triplet


class SoftRankingLoss(nn.Module):
    """
    Criterion that measures the soft ranking loss over a batch. It computes : softplus(input2 - target*input1)
    """
    def __init__(self, reduction='sum'):
        super(SoftRankingLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input1, input2, target):
        softplus = nn.Softplus()
        loss = softplus(input2.add(-target*input1))
        if self.reduction == 'sum':
            return torch.sum(loss)
        if self.reduction == 'mean':
            return torch.mean(loss)
