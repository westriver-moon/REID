#!/usr/bin/env python
"""Offline diagnostics for SYSU RGB/IR dual-encoder alignment experiments."""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.sysumm01.datasets.sysumm01 import (  # noqa: E402
    SYSUEvalDataset,
    build_test_records,
    l2_normalize,
)
from project.sysumm01.engine.evaluator import _extract_embeddings, _sample_gallery_indices  # noqa: E402
from project.sysumm01.models.reid_model import build_reid_model  # noqa: E402
from project.sysumm01.utils.config import load_config  # noqa: E402


DEFAULT_EXPERIMENTS = {
    "A0_no_adapter": "logs/sysu_rgb_ir_dual/dual_gated_a0_no_adapter_ep20",
    "A1_gated_h64": "logs/sysu_rgb_ir_dual/dual_gated_a1_h64_ep20",
    "A3_gated_h192": "logs/sysu_rgb_ir_dual/dual_gated_a3_h192_ep20",
    "A4_gated_h384": "logs/sysu_rgb_ir_dual/dual_gated_a4_h384_ep20",
}


def normalize_rows(array):
    return l2_normalize(np.asarray(array, dtype=np.float32))


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": None, "median": None, "std": None}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
    }


def load_checkpoint_model(exp_dir, device):
    config_path = exp_dir / "config.yaml"
    checkpoint_path = exp_dir / "checkpoints" / "best.pth"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    config = load_config(str(config_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    classifier_key = "classifier.weight"
    if classifier_key not in model_state:
        raise KeyError("Cannot infer num_classes: missing {}".format(classifier_key))
    num_classes = int(model_state[classifier_key].shape[0])
    model = build_reid_model(config["model"], num_classes=num_classes)
    model.load_state_dict(model_state, strict=True)
    model.to(device).eval()
    return model, config, checkpoint_path


def extract_eval_caches(model, config, device, mode="all"):
    dataset_root = config["eval"].get("dataset_root", config["dataset"].get("root"))
    image_size = tuple(config["dataset"]["image_size"])
    batch_size = int(config["eval"].get("batch_size", 128))
    num_workers = int(config["eval"].get("num_workers", 8))
    query_records, gallery_records = build_test_records(
        dataset_root,
        mode=mode,
        protocol=config["eval"].get("protocol", "cross_modality"),
    )
    query_cache = _extract_embeddings(
        model,
        query_records,
        image_size,
        batch_size,
        num_workers,
        device,
        dataset_root=dataset_root,
    )
    gallery_cache = _extract_embeddings(
        model,
        gallery_records,
        image_size,
        batch_size,
        num_workers,
        device,
        dataset_root=dataset_root,
    )
    return query_records, gallery_records, query_cache, gallery_cache


def valid_gallery_mask(query_camid, gallery_camids):
    adjusted_query_camid = 2 if int(query_camid) == 3 else int(query_camid)
    return gallery_camids != adjusted_query_camid


def analyze_trial(query_cache, gallery_cache, subset):
    q_feats = query_cache["features"]
    q_pids = query_cache["pids"].astype(np.int64)
    q_camids = query_cache["camids"].astype(np.int64)
    g_feats = gallery_cache["features"][subset]
    g_pids = gallery_cache["pids"][subset].astype(np.int64)
    g_camids = gallery_cache["camids"][subset].astype(np.int64)
    sims = q_feats @ g_feats.T

    positive_ranks = []
    hard_pos_sims = []
    hard_neg_sims = []
    hard_gaps = []
    top_hits = {1: [], 5: [], 10: [], 20: []}
    purities = {5: [], 10: [], 20: []}
    per_query = []

    for query_index in range(sims.shape[0]):
        valid = valid_gallery_mask(q_camids[query_index], g_camids)
        valid_indices = np.flatnonzero(valid)
        scores = sims[query_index, valid_indices]
        ranked_valid = valid_indices[np.argsort(-scores)]
        ranked_pids = g_pids[ranked_valid]
        matches = ranked_pids == q_pids[query_index]
        match_positions = np.flatnonzero(matches)
        if match_positions.size == 0:
            continue
        first_rank = int(match_positions[0] + 1)
        positive_ranks.append(first_rank)
        for topk in top_hits:
            top_hits[topk].append(float(first_rank <= topk))
        for topk in purities:
            k = min(topk, ranked_pids.shape[0])
            purities[topk].append(float(np.mean(ranked_pids[:k] == q_pids[query_index])))

        pos_mask = (g_pids == q_pids[query_index]) & valid
        neg_mask = (g_pids != q_pids[query_index]) & valid
        hard_pos = float(sims[query_index, pos_mask].min())
        hard_neg = float(sims[query_index, neg_mask].max())
        hard_pos_sims.append(hard_pos)
        hard_neg_sims.append(hard_neg)
        hard_gaps.append(hard_pos - hard_neg)

        ranks = match_positions + 1
        ap = float(np.mean(np.arange(1, ranks.size + 1) / ranks))
        per_query.append(
            {
                "pid": int(q_pids[query_index]),
                "positive_rank": first_rank,
                "rank1": float(first_rank == 1),
                "ap": ap,
                "hard_pos_sim": hard_pos,
                "hard_neg_sim": hard_neg,
                "hard_gap": hard_pos - hard_neg,
            }
        )

    metrics = {
        "positive_rank_mean": float(np.mean(positive_ranks)),
        "positive_rank_median": float(np.median(positive_ranks)),
        "top1_rate": float(np.mean(top_hits[1])),
        "top5_rate": float(np.mean(top_hits[5])),
        "top10_rate": float(np.mean(top_hits[10])),
        "top20_rate": float(np.mean(top_hits[20])),
        "hard_pos_sim_mean": float(np.mean(hard_pos_sims)),
        "hard_neg_sim_mean": float(np.mean(hard_neg_sims)),
        "hard_gap_mean": float(np.mean(hard_gaps)),
        "top5_purity": float(np.mean(purities[5])),
        "top10_purity": float(np.mean(purities[10])),
        "top20_purity": float(np.mean(purities[20])),
        "num_queries": int(len(positive_ranks)),
    }
    return metrics, per_query


def aggregate_trial_metrics(trial_outputs):
    keys = trial_outputs[0][0].keys()
    metrics = {}
    for key in keys:
        values = [item[0][key] for item in trial_outputs]
        metrics[key] = float(np.mean(values))

    by_pid = defaultdict(list)
    for _, per_query in trial_outputs:
        for item in per_query:
            by_pid[item["pid"]].append(item)

    per_id_rows = []
    for pid, rows in by_pid.items():
        ranks = [row["positive_rank"] for row in rows]
        aps = [row["ap"] for row in rows]
        rank1s = [row["rank1"] for row in rows]
        hard_gaps = [row["hard_gap"] for row in rows]
        per_id_rows.append(
            {
                "pid": int(pid),
                "query_trials": int(len(rows)),
                "rank1_rate": float(np.mean(rank1s)),
                "ap": float(np.mean(aps)),
                "positive_rank_median": float(np.median(ranks)),
                "positive_rank_mean": float(np.mean(ranks)),
                "hard_gap_mean": float(np.mean(hard_gaps)),
            }
        )
    per_id_rows.sort(key=lambda row: (row["rank1_rate"], row["ap"], -row["positive_rank_median"]))
    return metrics, per_id_rows


def analyze_retrieval(query_cache, gallery_cache, gallery_records, num_trials, seed):
    trials = _sample_gallery_indices(gallery_records, num_trials=num_trials, seed=seed)
    outputs = [analyze_trial(query_cache, gallery_cache, subset) for subset in trials]
    return aggregate_trial_metrics(outputs)


def analyze_prototypes(query_cache, gallery_cache):
    rgb_feats = gallery_cache["features"]
    rgb_pids = gallery_cache["pids"].astype(np.int64)
    ir_feats = query_cache["features"]
    ir_pids = query_cache["pids"].astype(np.int64)
    common = sorted(set(rgb_pids.tolist()) & set(ir_pids.tolist()))

    proto_dist = []
    proto_cos = []
    rgb_radius = []
    ir_radius = []
    cross_radius = []
    for pid in common:
        rf = rgb_feats[rgb_pids == pid]
        inf = ir_feats[ir_pids == pid]
        if rf.size == 0 or inf.size == 0:
            continue
        rc = normalize_rows(rf.mean(axis=0, keepdims=True))[0]
        ic = normalize_rows(inf.mean(axis=0, keepdims=True))[0]
        cc = normalize_rows(np.concatenate([rf, inf], axis=0).mean(axis=0, keepdims=True))[0]
        proto_dist.append(float(np.linalg.norm(rc - ic)))
        proto_cos.append(float(np.dot(rc, ic)))
        rgb_radius.append(float(np.linalg.norm(rf - rc[None, :], axis=1).mean()))
        ir_radius.append(float(np.linalg.norm(inf - ic[None, :], axis=1).mean()))
        cross_radius.append(float(np.linalg.norm(np.concatenate([rf, inf], axis=0) - cc[None, :], axis=1).mean()))

    return {
        "num_ids": int(len(proto_dist)),
        "prototype_distance": summarize(proto_dist),
        "prototype_cosine": summarize(proto_cos),
        "rgb_intra_radius": summarize(rgb_radius),
        "ir_intra_radius": summarize(ir_radius),
        "cross_intra_radius": summarize(cross_radius),
    }


def centroid_modality_accuracy(features, modalities):
    features = normalize_rows(features)
    modalities = np.asarray(modalities, dtype=np.int64)
    rgb_center = normalize_rows(features[modalities == 0].mean(axis=0, keepdims=True))[0]
    ir_center = normalize_rows(features[modalities == 1].mean(axis=0, keepdims=True))[0]
    rgb_score = features @ rgb_center
    ir_score = features @ ir_center
    pred = (ir_score > rgb_score).astype(np.int64)
    return float(np.mean(pred == modalities))


def layer_cross_modal_proxy(features, pids, modalities):
    features = normalize_rows(features)
    pids = np.asarray(pids, dtype=np.int64)
    modalities = np.asarray(modalities, dtype=np.int64)
    rgb = modalities == 0
    ir = modalities == 1
    rgb_feats = features[rgb]
    ir_feats = features[ir]
    rgb_pids = pids[rgb]
    ir_pids = pids[ir]
    sims = ir_feats @ rgb_feats.T

    pos_ranks = []
    hard_pos = []
    hard_neg = []
    hard_gap = []
    top1 = []
    for idx in range(sims.shape[0]):
        order = np.argsort(-sims[idx])
        ranked_pids = rgb_pids[order]
        matches = ranked_pids == ir_pids[idx]
        match_positions = np.flatnonzero(matches)
        if match_positions.size == 0:
            continue
        pos_ranks.append(int(match_positions[0] + 1))
        top1.append(float(match_positions[0] == 0))
        pos = sims[idx, rgb_pids == ir_pids[idx]]
        neg = sims[idx, rgb_pids != ir_pids[idx]]
        hard_pos.append(float(pos.min()))
        hard_neg.append(float(neg.max()))
        hard_gap.append(float(pos.min() - neg.max()))

    return {
        "positive_rank_median": float(np.median(pos_ranks)),
        "positive_rank_mean": float(np.mean(pos_ranks)),
        "top1_proxy": float(np.mean(top1)),
        "hard_pos_sim_mean": float(np.mean(hard_pos)),
        "hard_neg_sim_mean": float(np.mean(hard_neg)),
        "hard_gap_mean": float(np.mean(hard_gap)),
        "modality_centroid_acc": centroid_modality_accuracy(features, modalities),
    }


def extract_dual_layers(model, records, config, device, max_samples_per_modality=1200):
    dataset_root = config["eval"].get("dataset_root", config["dataset"].get("root"))
    image_size = tuple(config["dataset"]["image_size"])
    dataset = SYSUEvalDataset(records=records, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=int(config["eval"].get("batch_size", 128)),
        shuffle=False,
        num_workers=int(config["eval"].get("num_workers", 8)),
        pin_memory=True,
    )
    layer_features = defaultdict(list)
    pids = []
    modalities = []
    counts = {0: 0, 1: 0}

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_modalities = batch["modality"].numpy().astype(np.int64)
            keep = []
            for index, modality in enumerate(batch_modalities):
                if counts[int(modality)] < max_samples_per_modality:
                    keep.append(index)
                    counts[int(modality)] += 1
            if not keep:
                if min(counts.values()) >= max_samples_per_modality:
                    break
                continue
            keep = torch.as_tensor(keep, dtype=torch.long)
            images = batch["image"][keep].to(device, non_blocking=True)
            mods = batch["modality"][keep].to(device, non_blocking=True)
            batch_pids = batch["pid"][keep].numpy().astype(np.int64)
            batch_mods = mods.cpu().numpy().astype(np.int64)

            raw = images.new_zeros((images.shape[0], model.rgb_backbone.feature_dim))
            bn = images.new_zeros((images.shape[0], model.rgb_backbone.feature_dim))
            adapted = images.new_zeros((images.shape[0], model.rgb_backbone.feature_dim))
            projected = images.new_zeros((images.shape[0], model.feature_dim))
            for modality_id in (0, 1):
                mask = mods == modality_id
                if not mask.any():
                    continue
                if modality_id == 0:
                    backbone_out = model.rgb_backbone(images[mask])
                    branch_raw = backbone_out["features"]
                    branch_bn = model.rgb_bnneck(branch_raw)
                    branch_adapted = model.rgb_adapter(branch_bn)
                else:
                    backbone_out = model.ir_backbone(images[mask])
                    branch_raw = backbone_out["features"]
                    branch_bn = model.ir_bnneck(branch_raw)
                    branch_adapted = model.ir_adapter(branch_bn)
                branch_projected = model.shared_projector(branch_adapted)
                raw[mask] = branch_raw
                bn[mask] = branch_bn
                adapted[mask] = branch_adapted
                projected[mask] = branch_projected

            layer_features["encoder_raw"].append(raw.cpu().numpy())
            layer_features["bnneck"].append(bn.cpu().numpy())
            layer_features["adapter_out"].append(adapted.cpu().numpy())
            layer_features["projected"].append(projected.cpu().numpy())
            pids.append(batch_pids)
            modalities.append(batch_mods)
            if min(counts.values()) >= max_samples_per_modality:
                break

    merged = {key: np.concatenate(value, axis=0) for key, value in layer_features.items()}
    return merged, np.concatenate(pids, axis=0), np.concatenate(modalities, axis=0)


def pca_2d(features):
    features = np.asarray(features, dtype=np.float32)
    centered = features - features.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def save_pca_plots(name, query_cache, gallery_cache, output_dir, max_ids=20, per_id_per_modality=8):
    rng = np.random.default_rng(42)
    common = sorted(set(query_cache["pids"].tolist()) & set(gallery_cache["pids"].tolist()))
    if len(common) > max_ids:
        common = sorted(rng.choice(common, size=max_ids, replace=False).tolist())
    features = []
    labels = []
    modalities = []
    for pid in common:
        for cache, modality in ((gallery_cache, 0), (query_cache, 1)):
            indices = np.flatnonzero(cache["pids"] == pid)
            if indices.size > per_id_per_modality:
                indices = rng.choice(indices, size=per_id_per_modality, replace=False)
            features.append(cache["features"][indices])
            labels.extend([pid] * len(indices))
            modalities.extend([modality] * len(indices))
    features = np.concatenate(features, axis=0)
    labels = np.asarray(labels)
    modalities = np.asarray(modalities)
    xy = pca_2d(features)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = np.where(modalities == 0, "#1f77b4", "#d62728")
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=18, alpha=0.75)
    ax.set_title("{} PCA by modality (blue=RGB, red=IR)".format(name))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_dir / "{}_pca_modality.png".format(name), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    unique_ids = sorted(set(labels.tolist()))
    cmap = plt.get_cmap("tab20", len(unique_ids))
    id_to_color = {pid: cmap(index) for index, pid in enumerate(unique_ids)}
    ax.scatter(xy[:, 0], xy[:, 1], c=[id_to_color[pid] for pid in labels], s=18, alpha=0.75)
    ax.set_title("{} PCA by ID".format(name))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_dir / "{}_pca_id.png".format(name), dpi=180)
    plt.close(fig)


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_percent(value):
    return "{:.2f}".format(float(value) * 100.0)


def write_report(output_dir, summary_rows, prototype_summary, layer_summary):
    lines = [
        "# Dual Encoder Alignment 失败原因离线验证",
        "",
        "日期：2026-06-10",
        "",
        "说明：本报告只基于已有 checkpoint 和 SYSU-MM01 eval split 生成，没有重新训练。positive rank / hard pair 按 SYSU cross-modality 的 gallery trial 和相机过滤规则计算。",
        "",
        "## 1. 检索排序诊断",
        "",
        "| 实验 | Top-1% | Top-5% | Top-10% | Top-20% | positive rank median | hard gap | Top-10 purity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {name} | {top1} | {top5} | {top10} | {top20} | {median:.1f} | {hard_gap:.4f} | {purity:.4f} |".format(
                name=row["experiment"],
                top1=format_percent(row["top1_rate"]),
                top5=format_percent(row["top5_rate"]),
                top10=format_percent(row["top10_rate"]),
                top20=format_percent(row["top20_rate"]),
                median=row["positive_rank_median"],
                hard_gap=row["hard_gap_mean"],
                purity=row["top10_purity"],
            )
        )

    lines.extend(
        [
            "",
            "## 2. Prototype / 类内半径",
            "",
            "| 实验 | prototype distance | prototype cosine | RGB radius | IR radius | cross radius |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, item in prototype_summary.items():
        lines.append(
            "| {name} | {pd:.4f} | {pc:.4f} | {rr:.4f} | {ir:.4f} | {cr:.4f} |".format(
                name=name,
                pd=item["prototype_distance"]["mean"],
                pc=item["prototype_cosine"]["mean"],
                rr=item["rgb_intra_radius"]["mean"],
                ir=item["ir_intra_radius"]["mean"],
                cr=item["cross_intra_radius"]["mean"],
            )
        )

    lines.extend(
        [
            "",
            "## 3. Adapter 前后特征",
            "",
            "| 实验 | 层 | Top-1 proxy% | positive rank median | hard gap | modality centroid acc% |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for name, layers in layer_summary.items():
        for layer, item in layers.items():
            lines.append(
                "| {name} | {layer} | {top1} | {rank:.1f} | {gap:.4f} | {macc} |".format(
                    name=name,
                    layer=layer,
                    top1=format_percent(item["top1_proxy"]),
                    rank=item["positive_rank_median"],
                    gap=item["hard_gap_mean"],
                    macc=format_percent(item["modality_centroid_acc"]),
                )
            )

    lines.extend(
        [
            "",
            "## 4. 输出文件",
            "",
            "- `summary_metrics.csv`：positive rank、hard pair、top-k purity 汇总。",
            "- `prototype_metrics.json`：prototype distance 和类内半径。",
            "- `layer_metrics.json`：adapter 前后分层指标。",
            "- `per_id/*.csv`：每个实验的 hard identity 排序。",
            "- `*_pca_modality.png` / `*_pca_id.png`：PCA 可视化。服务器当前缺少 sklearn，因此这里使用 PCA 作为轻量替代，而不是 t-SNE/UMAP。",
            "",
        ]
    )
    with open(output_dir / "analysis_report.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--output-dir", default="logs/sysu_rgb_ir_dual/alignment_analysis")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--mode", default="all")
    parser.add_argument("--layer-sample-per-modality", type=int, default=1200)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "per_id").mkdir(exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    summary_rows = []
    full_summary = {}
    prototype_summary = {}
    layer_summary = {}

    for name, rel_dir in DEFAULT_EXPERIMENTS.items():
        exp_dir = repo_root / rel_dir
        print("Analyzing {} from {}".format(name, exp_dir), flush=True)
        model, config, checkpoint_path = load_checkpoint_model(exp_dir, device)
        query_records, gallery_records, query_cache, gallery_cache = extract_eval_caches(
            model,
            config,
            device,
            mode=args.mode,
        )
        metrics, per_id_rows = analyze_retrieval(
            query_cache,
            gallery_cache,
            gallery_records,
            num_trials=args.num_trials,
            seed=int(config.get("seed", 42)),
        )
        proto = analyze_prototypes(query_cache, gallery_cache)
        save_pca_plots(name, query_cache, gallery_cache, output_dir)

        summary_row = {"experiment": name, "checkpoint": str(checkpoint_path)}
        summary_row.update(metrics)
        summary_rows.append(summary_row)
        full_summary[name] = {"retrieval": metrics, "prototype": proto}
        prototype_summary[name] = proto

        write_csv(
            output_dir / "per_id" / "{}_per_id.csv".format(name),
            per_id_rows,
            [
                "pid",
                "query_trials",
                "rank1_rate",
                "ap",
                "positive_rank_median",
                "positive_rank_mean",
                "hard_gap_mean",
            ],
        )

        if name in ("A0_no_adapter", "A3_gated_h192", "A4_gated_h384"):
            combined_records = list(gallery_records) + list(query_records)
            layers, pids, modalities = extract_dual_layers(
                model,
                combined_records,
                config,
                device,
                max_samples_per_modality=args.layer_sample_per_modality,
            )
            layer_summary[name] = {
                layer_name: layer_cross_modal_proxy(features, pids, modalities)
                for layer_name, features in layers.items()
            }

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(
        output_dir / "summary_metrics.csv",
        summary_rows,
        [
            "experiment",
            "checkpoint",
            "positive_rank_mean",
            "positive_rank_median",
            "top1_rate",
            "top5_rate",
            "top10_rate",
            "top20_rate",
            "hard_pos_sim_mean",
            "hard_neg_sim_mean",
            "hard_gap_mean",
            "top5_purity",
            "top10_purity",
            "top20_purity",
            "num_queries",
        ],
    )
    with open(output_dir / "summary_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(full_summary, handle, indent=2, sort_keys=True)
    with open(output_dir / "prototype_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(prototype_summary, handle, indent=2, sort_keys=True)
    with open(output_dir / "layer_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(layer_summary, handle, indent=2, sort_keys=True)
    write_report(output_dir, summary_rows, prototype_summary, layer_summary)
    print("Wrote diagnostics to {}".format(output_dir), flush=True)


if __name__ == "__main__":
    main()
