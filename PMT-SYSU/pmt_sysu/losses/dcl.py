from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def pdist_torch(emb1, emb2):
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx.addmm_(emb1, emb2.t(), beta=1, alpha=-2)
    return dist_mtx.clamp(min=1e-12).sqrt()


class DCL(nn.Module):
    def __init__(self, num_pos=4, feat_norm="no"):
        super().__init__()
        self.num_pos = int(num_pos)
        self.feat_norm = feat_norm

    def forward(self, inputs, targets):
        if self.feat_norm == "yes":
            inputs = F.normalize(inputs, p=2, dim=-1)

        total = inputs.size(0)
        assert total % (2 * self.num_pos) == 0, "DCL expects [visible, ir] with fixed num_pos chunks"
        left, right = targets.chunk(2, 0)
        assert torch.equal(left, right), "DCL requires aligned visible and IR labels"
        id_num = total // 2 // self.num_pos

        is_neg = targets.expand(total, total).ne(targets.expand(total, total).t())
        is_neg_c2i = is_neg[:: self.num_pos, :].chunk(2, 0)[0]

        centers = []
        for i in range(id_num):
            centers.append(inputs[targets == targets[i * self.num_pos]].mean(0))
        centers = torch.stack(centers)

        dist_mat = pdist_torch(centers, inputs)
        an = dist_mat * is_neg_c2i
        an = an[an > 1e-6].view(id_num, -1)
        d_neg = torch.mean(an, dim=1, keepdim=True)
        mask_an = (an - d_neg).expand(id_num, total - 2 * self.num_pos).lt(0)
        an = an * mask_an

        list_an = []
        for i in range(id_num):
            values = an[i][an[i] > 1e-6]
            list_an.append(torch.mean(values) if values.numel() else d_neg[i, 0])
        an_mean = sum(list_an) / len(list_an)

        ap = dist_mat * ~is_neg_c2i
        ap_mean = torch.mean(ap[ap > 1e-6])
        return ap_mean / (an_mean + 1e-12)

