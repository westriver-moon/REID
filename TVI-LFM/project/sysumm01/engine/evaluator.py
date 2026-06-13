import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from project.sysumm01.datasets.sysumm01 import SYSUEvalDataset, build_test_records, l2_normalize


def _stable_unique(array):
    _, indices = np.unique(array, return_index=True)
    return array[np.sort(indices)]


def _compute_cmc(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    gallery_unique_count = _stable_unique(gallery_ids).shape[0]
    match_counter = np.zeros((gallery_unique_count,), dtype=np.float64)
    ranked_ids = gallery_ids[sorted_indices]
    ranked_cams = gallery_cam_ids[sorted_indices]

    valid = 0
    for probe_index in range(sorted_indices.shape[0]):
        result = ranked_ids[probe_index].copy()
        result[ranked_cams[probe_index] == query_cam_ids[probe_index]] = -1
        result = np.array([item for item in result if item != -1])
        result = _stable_unique(result)
        matches = np.equal(result, query_ids[probe_index])
        if np.sum(matches) > 0:
            valid += 1
            match_counter[: matches.shape[0]] += matches.astype(np.float64)

    if valid == 0:
        return np.zeros((gallery_unique_count,), dtype=np.float64)
    return np.cumsum(match_counter / valid)


def _compute_map(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    ranked_ids = gallery_ids[sorted_indices]
    ranked_cams = gallery_cam_ids[sorted_indices]
    avg_precision_sum = 0.0
    valid = 0

    for probe_index in range(sorted_indices.shape[0]):
        result = ranked_ids[probe_index].copy()
        result[ranked_cams[probe_index] == query_cam_ids[probe_index]] = -1
        result = np.array([item for item in result if item != -1])
        matches = result == query_ids[probe_index]
        true_match_count = np.sum(matches)
        if true_match_count == 0:
            continue
        valid += 1
        ranks = np.where(matches)[0]
        precisions = np.arange(1, true_match_count + 1) / (ranks + 1)
        avg_precision_sum += precisions.mean()

    if valid == 0:
        return 0.0
    return avg_precision_sum / valid


def _extract_embeddings(
    model,
    records,
    image_size,
    batch_size,
    num_workers,
    device,
    dataset_root,
    schp_mask_root=None,
    schp_min_part_pixels=4,
    schp_allow_fallback=True,
    schp_quality_index=None,
):
    dataset = SYSUEvalDataset(
        records=records,
        image_size=image_size,
        schp_mask_root=schp_mask_root,
        schp_source_root=dataset_root,
        schp_min_part_pixels=schp_min_part_pixels,
        schp_allow_fallback=schp_allow_fallback,
        schp_quality_index=schp_quality_index,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    features = []
    pids = []
    camids = []
    paths = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            part_masks = batch.get("part_masks")
            if part_masks is not None:
                part_masks = part_masks.to(device, non_blocking=True)
            modalities = batch.get("modality")
            if modalities is not None:
                modalities = modalities.to(device, non_blocking=True)
            embeddings = model.extract_features(images, part_masks=part_masks, modality=modalities)
            features.append(embeddings.cpu().numpy())
            pids.append(batch["pid"].numpy())
            camids.append(batch["camid"].numpy())
            paths.extend(batch["path"])

    return {
        "features": l2_normalize(np.concatenate(features, axis=0)),
        "pids": np.concatenate(pids, axis=0),
        "camids": np.concatenate(camids, axis=0),
        "paths": np.array(paths),
    }


def _sample_gallery_indices(records, num_trials, seed):
    by_pid_cam = {}
    for index, record in enumerate(records):
        key = (record["pid"], record["camid"])
        by_pid_cam.setdefault(key, []).append(index)

    person_ids = sorted({record["pid"] for record in records})
    camera_ids = sorted({record["camid"] for record in records})
    trials = []
    for trial in range(num_trials):
        rng = random.Random(seed + trial)
        indices = []
        for pid in person_ids:
            for camid in camera_ids:
                key = (pid, camid)
                if key not in by_pid_cam:
                    continue
                candidates = by_pid_cam[key]
                indices.append(rng.choice(candidates))
        trials.append(np.array(indices, dtype=np.int64))
    return trials


def evaluate_sysu(
    model,
    dataset_root,
    image_size,
    batch_size,
    num_workers,
    device,
    mode="all",
    num_trials=10,
    seed=42,
    protocol="cross_modality",
    modality=None,
    id_split="test",
    schp_mask_root=None,
    schp_min_part_pixels=4,
    schp_allow_fallback=True,
    schp_quality_index=None,
):
    query_records, gallery_records = build_test_records(
        dataset_root,
        mode=mode,
        protocol=protocol,
        modality=modality,
        split=id_split,
    )
    query_cache = _extract_embeddings(
        model,
        query_records,
        image_size,
        batch_size,
        num_workers,
        device,
        dataset_root=dataset_root,
        schp_mask_root=schp_mask_root,
        schp_min_part_pixels=schp_min_part_pixels,
        schp_allow_fallback=schp_allow_fallback,
        schp_quality_index=schp_quality_index,
    )
    gallery_cache = _extract_embeddings(
        model,
        gallery_records,
        image_size,
        batch_size,
        num_workers,
        device,
        dataset_root=dataset_root,
        schp_mask_root=schp_mask_root,
        schp_min_part_pixels=schp_min_part_pixels,
        schp_allow_fallback=schp_allow_fallback,
        schp_quality_index=schp_quality_index,
    )

    query_cam_ids = query_cache["camids"].copy()
    if protocol == "cross_modality":
        query_cam_ids[query_cam_ids == 3] = 2
    gallery_trials = _sample_gallery_indices(gallery_records, num_trials=num_trials, seed=seed)

    metric_sums = {"mAP": 0.0, "rank1": 0.0, "rank5": 0.0, "rank10": 0.0, "rank20": 0.0}
    retrieval_examples = []

    for trial_index, subset in enumerate(gallery_trials):
        gallery_feats = gallery_cache["features"][subset]
        gallery_ids = gallery_cache["pids"][subset]
        gallery_camids = gallery_cache["camids"][subset]
        gallery_paths = gallery_cache["paths"][subset]

        similarities = np.dot(query_cache["features"], gallery_feats.T)
        sorted_indices = np.argsort(-similarities, axis=1)

        cmc = _compute_cmc(
            sorted_indices,
            query_cache["pids"],
            query_cam_ids,
            gallery_ids,
            gallery_camids,
        )
        metric_sums["mAP"] += _compute_map(
            sorted_indices,
            query_cache["pids"],
            query_cam_ids,
            gallery_ids,
            gallery_camids,
        )
        metric_sums["rank1"] += cmc[0] if cmc.shape[0] > 0 else 0.0
        metric_sums["rank5"] += cmc[4] if cmc.shape[0] > 4 else cmc[-1]
        metric_sums["rank10"] += cmc[9] if cmc.shape[0] > 9 else cmc[-1]
        metric_sums["rank20"] += cmc[19] if cmc.shape[0] > 19 else cmc[-1]

        if trial_index == 0:
            for query_index in range(min(8, similarities.shape[0])):
                topk = sorted_indices[query_index, :5]
                retrieval_examples.append(
                    {
                        "query_path": query_cache["paths"][query_index],
                        "query_pid": int(query_cache["pids"][query_index]),
                        "query_camid": int(query_cache["camids"][query_index]),
                        "gallery_paths": gallery_paths[topk].tolist(),
                        "gallery_pids": gallery_ids[topk].astype(int).tolist(),
                        "scores": similarities[query_index, topk].astype(float).tolist(),
                    }
                )

    metrics = {}
    for key, value in metric_sums.items():
        metrics[key] = float(value / max(num_trials, 1))
    metrics["num_queries"] = int(len(query_records))
    metrics["num_gallery_candidates"] = int(len(gallery_records))
    metrics["mode"] = mode
    metrics["protocol"] = protocol
    metrics["modality"] = modality or "cross"
    metrics["id_split"] = id_split
    return metrics, retrieval_examples
