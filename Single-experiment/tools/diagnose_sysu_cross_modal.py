#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.sysumm01.datasets.sysumm01 import build_test_records
from project.sysumm01.engine.evaluator import _extract_embeddings, _sample_gallery_indices
from project.sysumm01.models.reid_model import build_reid_model
from project.sysumm01.utils.config import load_config


def _load_state_dict(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def _metric(values, fn, default=0.0):
    values = np.asarray(values)
    if values.size == 0:
        return float(default)
    return float(fn(values))


def _cmc_map(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    ranked_ids = gallery_ids[sorted_indices]
    ranked_cams = gallery_cam_ids[sorted_indices]
    rank_hits = {1: [], 5: [], 10: [], 20: []}
    aps = []
    positive_ranks = []
    topk_purity = {5: [], 10: [], 20: []}

    for q_index in range(sorted_indices.shape[0]):
        result_ids = ranked_ids[q_index].copy()
        result_cams = ranked_cams[q_index].copy()
        keep = result_cams != query_cam_ids[q_index]
        result_ids = result_ids[keep]
        matches = result_ids == query_ids[q_index]
        if not matches.any():
            continue
        first_rank = int(np.where(matches)[0][0] + 1)
        positive_ranks.append(first_rank)
        for k in rank_hits:
            rank_hits[k].append(float(matches[:k].any()))
            if k in topk_purity:
                topk_purity[k].append(float(matches[:k].mean()))
        ranks = np.where(matches)[0]
        precisions = np.arange(1, len(ranks) + 1) / (ranks + 1)
        aps.append(float(precisions.mean()))

    return {
        "mAP_proxy": _metric(aps, np.mean),
        "positive_rank_mean": _metric(positive_ranks, np.mean),
        "positive_rank_median": _metric(positive_ranks, np.median),
        "positive_rank_p75": _metric(positive_ranks, lambda x: np.percentile(x, 75)),
        "positive_rank_p90": _metric(positive_ranks, lambda x: np.percentile(x, 90)),
        "top1_rate": _metric(rank_hits[1], np.mean),
        "top5_rate": _metric(rank_hits[5], np.mean),
        "top10_rate": _metric(rank_hits[10], np.mean),
        "top20_rate": _metric(rank_hits[20], np.mean),
        "top5_purity": _metric(topk_purity[5], np.mean),
        "top10_purity": _metric(topk_purity[10], np.mean),
        "top20_purity": _metric(topk_purity[20], np.mean),
    }


def _hard_pair_stats(query_features, query_ids, gallery_features, gallery_ids):
    sim = query_features @ gallery_features.T
    positive = query_ids[:, None] == gallery_ids[None, :]
    negative = ~positive
    hard_pos = []
    hard_neg = []
    mean_pos = []
    mean_neg = []
    for index in range(sim.shape[0]):
        if not positive[index].any() or not negative[index].any():
            continue
        pos_scores = sim[index][positive[index]]
        neg_scores = sim[index][negative[index]]
        hard_pos.append(float(pos_scores.min()))
        hard_neg.append(float(neg_scores.max()))
        mean_pos.append(float(pos_scores.mean()))
        mean_neg.append(float(neg_scores.mean()))
    return {
        "mean_positive_sim": _metric(mean_pos, np.mean),
        "mean_negative_sim": _metric(mean_neg, np.mean),
        "mean_gap": _metric(np.asarray(mean_pos) - np.asarray(mean_neg), np.mean),
        "hard_positive_sim": _metric(hard_pos, np.mean),
        "hard_negative_sim": _metric(hard_neg, np.mean),
        "hard_gap": _metric(np.asarray(hard_pos) - np.asarray(hard_neg), np.mean),
    }


def _prototype_stats(query_features, query_ids, gallery_features, gallery_ids):
    common_ids = sorted(set(query_ids.tolist()) & set(gallery_ids.tolist()))
    proto_dist = []
    query_radius = []
    gallery_radius = []
    cross_radius = []
    for pid in common_ids:
        q = query_features[query_ids == pid]
        g = gallery_features[gallery_ids == pid]
        if len(q) == 0 or len(g) == 0:
            continue
        q_center = q.mean(axis=0)
        g_center = g.mean(axis=0)
        q_center = q_center / max(np.linalg.norm(q_center), 1e-12)
        g_center = g_center / max(np.linalg.norm(g_center), 1e-12)
        joint_center = np.concatenate([q, g], axis=0).mean(axis=0)
        joint_center = joint_center / max(np.linalg.norm(joint_center), 1e-12)
        proto_dist.append(float(np.linalg.norm(q_center - g_center)))
        query_radius.append(float(np.linalg.norm(q - q_center[None, :], axis=1).mean()))
        gallery_radius.append(float(np.linalg.norm(g - g_center[None, :], axis=1).mean()))
        cross_radius.append(float(np.linalg.norm(np.concatenate([q, g], axis=0) - joint_center[None, :], axis=1).mean()))
    return {
        "prototype_distance": _metric(proto_dist, np.mean),
        "query_intra_radius": _metric(query_radius, np.mean),
        "gallery_intra_radius": _metric(gallery_radius, np.mean),
        "cross_modal_intra_radius": _metric(cross_radius, np.mean),
    }


def _modality_centroid_accuracy(query_features, gallery_features):
    features = np.concatenate([query_features, gallery_features], axis=0)
    labels = np.concatenate([
        np.ones(len(query_features), dtype=np.int64),
        np.zeros(len(gallery_features), dtype=np.int64),
    ])
    rgb_center = gallery_features.mean(axis=0)
    ir_center = query_features.mean(axis=0)
    rgb_center = rgb_center / max(np.linalg.norm(rgb_center), 1e-12)
    ir_center = ir_center / max(np.linalg.norm(ir_center), 1e-12)
    rgb_sim = features @ rgb_center
    ir_sim = features @ ir_center
    pred = (ir_sim > rgb_sim).astype(np.int64)
    return float((pred == labels).mean())


def diagnose_mode(model, config, device, mode, num_trials):
    dataset_root = config["eval"].get("dataset_root", config["dataset"].get("root"))
    image_size = tuple(config["dataset"]["image_size"])
    query_records, gallery_records = build_test_records(
        dataset_root,
        mode=mode,
        protocol="cross_modality",
        modality=None,
    )
    query_cache = _extract_embeddings(
        model,
        query_records,
        image_size,
        config["eval"]["batch_size"],
        config["eval"]["num_workers"],
        device,
        dataset_root=dataset_root,
    )
    gallery_cache = _extract_embeddings(
        model,
        gallery_records,
        image_size,
        config["eval"]["batch_size"],
        config["eval"]["num_workers"],
        device,
        dataset_root=dataset_root,
    )

    query_cam_ids = query_cache["camids"].copy()
    query_cam_ids[query_cam_ids == 3] = 2
    trials = _sample_gallery_indices(gallery_records, num_trials=num_trials, seed=config["seed"])

    rank_stats = []
    hard_stats = []
    proto_stats = []
    modality_acc = []
    for subset in trials:
        gallery_features = gallery_cache["features"][subset]
        gallery_ids = gallery_cache["pids"][subset]
        gallery_camids = gallery_cache["camids"][subset]
        sim = query_cache["features"] @ gallery_features.T
        sorted_indices = np.argsort(-sim, axis=1)
        rank_stats.append(
            _cmc_map(
                sorted_indices,
                query_cache["pids"],
                query_cam_ids,
                gallery_ids,
                gallery_camids,
            )
        )
        hard_stats.append(_hard_pair_stats(query_cache["features"], query_cache["pids"], gallery_features, gallery_ids))
        proto_stats.append(_prototype_stats(query_cache["features"], query_cache["pids"], gallery_features, gallery_ids))
        modality_acc.append(_modality_centroid_accuracy(query_cache["features"], gallery_features))

    def avg_dict(dicts):
        keys = dicts[0].keys()
        return {key: float(np.mean([item[key] for item in dicts])) for key in keys}

    return {
        "mode": mode,
        "num_trials": num_trials,
        "num_queries": int(len(query_records)),
        "num_gallery_candidates": int(len(gallery_records)),
        "rank": avg_dict(rank_stats),
        "hard_pairs": avg_dict(hard_stats),
        "prototype": avg_dict(proto_stats),
        "modality_centroid_acc": float(np.mean(modality_acc)),
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose SYSU cross-modal embedding geometry.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--modes", nargs="+", default=["all", "indoor"])
    args = parser.parse_args()

    config = load_config(args.config)
    config["model"]["image_size"] = list(config["dataset"]["image_size"])
    num_classes = int(config["dataset"].get("num_classes", 395))
    model = build_reid_model(config["model"], num_classes=num_classes)
    model.load_state_dict(_load_state_dict(args.checkpoint), strict=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    result = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "trials": args.trials,
        "modes": {},
    }
    for mode in args.modes:
        print("Diagnosing mode={}...".format(mode), flush=True)
        result["modes"][mode] = diagnose_mode(model, config, device, mode, args.trials)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote {}".format(output), flush=True)


if __name__ == "__main__":
    main()
