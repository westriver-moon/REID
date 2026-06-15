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


class MSEL(nn.Module):
    def __init__(self, num_pos, feat_norm="no"):
        super().__init__()
        self.num_pos = int(num_pos)
        self.feat_norm = feat_norm

    def forward(self, inputs, targets):
        if self.feat_norm == "yes":
            inputs = F.normalize(inputs, p=2, dim=-1)

        assert inputs.size(0) % 2 == 0, "MSEL expects [visible, ir] concatenation"
        target, target_ir = targets.chunk(2, 0)
        assert torch.equal(target, target_ir), "MSEL requires aligned visible and IR labels"
        num = target.size(0)
        dist_mat = pdist_torch(inputs, inputs)

        dist_intra_rgb = dist_mat[0:num, 0:num]
        dist_cross_rgb = dist_mat[0:num, num : 2 * num]
        dist_intra_ir = dist_mat[num : 2 * num, num : 2 * num]
        dist_cross_ir = dist_mat[num : 2 * num, 0:num]

        is_pos = target.expand(num, num).eq(target.expand(num, num).t())

        intra_rgb, _ = (is_pos * dist_intra_rgb).topk(self.num_pos - 1, dim=1, largest=True, sorted=False)
        intra_mean_rgb = torch.mean(intra_rgb, dim=1)

        intra_ir, _ = (is_pos * dist_intra_ir).topk(self.num_pos - 1, dim=1, largest=True, sorted=False)
        intra_mean_ir = torch.mean(intra_ir, dim=1)

        dist_cross_rgb = dist_cross_rgb[is_pos].contiguous().view(num, -1)
        cross_mean_rgb = torch.mean(dist_cross_rgb, dim=1)

        dist_cross_ir = dist_cross_ir[is_pos].contiguous().view(num, -1)
        cross_mean_ir = torch.mean(dist_cross_ir, dim=1)

        loss = (
            torch.mean(torch.pow(cross_mean_rgb - intra_mean_rgb, 2))
            + torch.mean(torch.pow(cross_mean_ir - intra_mean_ir, 2))
        ) / 2
        return loss

