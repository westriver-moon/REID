from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pmt_sysu.data.dataset import TestData
from pmt_sysu.data.sysu_protocol import process_gallery_sysu, process_query_sysu
from pmt_sysu.data.transforms import build_transforms
from pmt_sysu.utils.metrics import eval_sysu


@torch.no_grad()
def extract_features(model, loader, device, feature_dim: int = 768):
    model.eval()
    features = np.zeros((len(loader.dataset), feature_dim), dtype=np.float32)
    ptr = 0
    for inputs, _labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        batch = inputs.size(0)
        feats = model(inputs)
        features[ptr : ptr + batch] = feats.detach().cpu().numpy()
        ptr += batch
    return features


def evaluate_sysu(
    model,
    data_root,
    height: int,
    width: int,
    mode: str = "all",
    gallery_mode: str = "single",
    trials: int = 10,
    batch_size: int = 128,
    num_workers: int = 4,
    device: str | torch.device = "cuda",
    output_dir: str | Path | None = None,
    logger=print,
):
    transforms = build_transforms(height, width)
    query_img, query_label, query_cam = process_query_sysu(data_root, mode=mode)
    query_set = TestData(query_img, query_label, transform=transforms["test"], img_size=(width, height))
    query_loader = DataLoader(query_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    query_feat = extract_features(model, query_loader, device)

    output_dir = Path(output_dir) if output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    all_cmc = None
    all_map = 0.0
    all_minp = 0.0
    trial_results = []
    for trial in tqdm(range(trials), desc="SYSU trials"):
        gall_img, gall_label, gall_cam = process_gallery_sysu(
            data_root, mode=mode, trial=trial, gall_mode=gallery_mode
        )
        gall_set = TestData(gall_img, gall_label, transform=transforms["test"], img_size=(width, height))
        gall_loader = DataLoader(gall_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        gall_feat = extract_features(model, gall_loader, device)
        distmat = -np.matmul(query_feat, gall_feat.T)
        cmc, mAP, mINP = eval_sysu(distmat, query_label, gall_label, query_cam, gall_cam)
        result = {
            "trial": trial,
            "rank1": float(cmc[0]),
            "rank5": float(cmc[4]),
            "rank10": float(cmc[9]),
            "rank20": float(cmc[19]),
            "mAP": float(mAP),
            "mINP": float(mINP),
        }
        trial_results.append(result)
        if output_dir:
            (output_dir / f"trial_{trial:02d}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        all_cmc = cmc.copy() if all_cmc is None else all_cmc + cmc
        all_map += mAP
        all_minp += mINP
        logger(
            f"trial {trial:02d}: mAP={mAP:.4f} mINP={mINP:.4f} "
            f"R1={cmc[0]:.4f} R5={cmc[4]:.4f} R10={cmc[9]:.4f} R20={cmc[19]:.4f}"
        )

    avg_cmc = all_cmc / trials
    average = {
        "mode": mode,
        "gallery_mode": gallery_mode,
        "trials": trials,
        "rank1": float(avg_cmc[0]),
        "rank5": float(avg_cmc[4]),
        "rank10": float(avg_cmc[9]),
        "rank20": float(avg_cmc[19]),
        "mAP": float(all_map / trials),
        "mINP": float(all_minp / trials),
    }
    if output_dir:
        (output_dir / "average.json").write_text(json.dumps(average, indent=2), encoding="utf-8")
    logger(
        "average: "
        f"mAP={average['mAP']:.4f} mINP={average['mINP']:.4f} "
        f"R1={average['rank1']:.4f} R5={average['rank5']:.4f} "
        f"R10={average['rank10']:.4f} R20={average['rank20']:.4f}"
    )
    return average, trial_results

