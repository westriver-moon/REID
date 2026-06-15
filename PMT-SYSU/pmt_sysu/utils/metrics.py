from __future__ import annotations

import numpy as np


def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20):
    """Official SYSU metric logic from PMT with same-camera filtering."""
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g

    indices = np.argsort(distmat, axis=1)
    pred_label = g_pids[indices]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    new_all_cmc = []
    all_cmc = []
    all_ap = []
    all_inp = []
    num_valid_q = 0.0

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]
        remove = (q_camid == 3) & (g_camids[order] == 2)
        keep = np.invert(remove)

        new_cmc = pred_label[q_idx][keep]
        new_index = np.unique(new_cmc, return_index=True)[1]
        new_cmc = [new_cmc[index] for index in sorted(new_index)]
        new_match = (new_cmc == q_pid).astype(np.int32)
        new_cmc = new_match.cumsum()
        new_all_cmc.append(new_cmc[:max_rank])

        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue
        cmc = orig_cmc.cumsum()
        pos_idx = np.where(orig_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx] / (pos_max_idx + 1.0)
        all_inp.append(inp)
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = np.asarray([x / (i + 1.0) for i, x in enumerate(tmp_cmc)]) * orig_cmc
        all_ap.append(tmp_cmc.sum() / num_rel)

    assert num_valid_q > 0, "all query identities do not appear in gallery"
    all_cmc = np.asarray(all_cmc).astype(np.float32).sum(0) / num_valid_q
    new_all_cmc = np.asarray(new_all_cmc).astype(np.float32).sum(0) / num_valid_q
    return new_all_cmc, float(np.mean(all_ap)), float(np.mean(all_inp))

