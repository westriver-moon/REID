from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def euclidean_dist(x, y, eps=1e-12):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    return dist.clamp(min=eps).sqrt()


def hard_example_mining(dist_mat, target):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    num = dist_mat.size(0)
    is_pos = target.expand(num, num).eq(target.expand(num, num).t())
    is_neg = target.expand(num, num).ne(target.expand(num, num).t())
    dist_ap, _ = torch.max(dist_mat[is_pos].contiguous().view(num, -1), 1, keepdim=True)
    dist_an, _ = torch.min(dist_mat[is_neg].contiguous().view(num, -1), 1, keepdim=True)
    return dist_ap.squeeze(1), dist_an.squeeze(1)


class TripletLoss(nn.Module):
    def __init__(self, margin, feat_norm="yes"):
        super().__init__()
        self.margin = margin
        self.feat_norm = feat_norm
        self.ranking_loss = nn.MarginRankingLoss(margin=margin) if margin >= 0 else nn.SoftMarginLoss()

    def forward(self, global_feat1, global_feat2, target):
        if self.feat_norm == "yes":
            global_feat1 = F.normalize(global_feat1, p=2, dim=-1)
            global_feat2 = F.normalize(global_feat2, p=2, dim=-1)
        dist_mat = euclidean_dist(global_feat1, global_feat2)
        dist_ap, dist_an = hard_example_mining(dist_mat, target)
        y = dist_an.new_ones(dist_an.size())
        if self.margin >= 0:
            return self.ranking_loss(dist_an, dist_ap, y)
        return self.ranking_loss(dist_an - dist_ap, y)

