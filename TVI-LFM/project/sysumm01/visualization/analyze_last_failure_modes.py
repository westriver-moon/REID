import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.sysumm01.datasets.sysumm01 import build_test_records, build_transforms
from project.sysumm01.models.reid_model import ReIDModel
from project.sysumm01.utils.config import dump_json, load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze LAST failure modes on SYSU-MM01")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--mode", default="all", choices=["all", "indoor"])
    parser.add_argument("--num-pairs", type=int, default=256)
    parser.add_argument("--topk-patches", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pair-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sample_pairs(dataset_root, mode, num_pairs, seed):
    query_records, gallery_records = build_test_records(dataset_root, mode=mode)
    query_by_pid = {}
    gallery_by_pid = {}
    for record in query_records:
        query_by_pid.setdefault(record["pid"], []).append(record)
    for record in gallery_records:
        gallery_by_pid.setdefault(record["pid"], []).append(record)

    candidate_pids = sorted(set(query_by_pid) & set(gallery_by_pid))
    if not candidate_pids:
        raise ValueError("No cross-modal pairs available")

    rng = random.Random(seed)
    pairs = []
    for _ in range(num_pairs):
        pid = rng.choice(candidate_pids)
        pairs.append(
            {
                "pid": pid,
                "query": rng.choice(query_by_pid[pid]),
                "gallery": rng.choice(gallery_by_pid[pid]),
            }
        )
    return pairs


def load_batch(records, transform):
    tensors = []
    for record in records:
        image = Image.open(record["path"]).convert("RGB")
        tensors.append(transform(image))
    return torch.stack(tensors, dim=0)


@torch.no_grad()
def forward_embeddings_and_scores(model, images):
    backbone_outputs = model.backbone(images)
    bn_feat = model.bnneck(backbone_outputs["features"])
    embeddings = F.normalize(bn_feat, dim=1)
    return embeddings, backbone_outputs["patch_scores"]


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()) if array.size else 0.0,
        "std": float(array.std()) if array.size else 0.0,
        "min": float(array.min()) if array.size else 0.0,
        "max": float(array.max()) if array.size else 0.0,
    }


def compute_metrics(model, pairs, image_size, batch_size, topk_patches, device):
    transform = build_transforms(image_size=image_size, training=False)
    grid_h, grid_w = model.backbone.vit.patch_embed.grid_size

    pair_cosines = []
    overlap_ratios = []
    rgb_entropies = []
    ir_entropies = []
    rgb_top1_mass = []
    ir_top1_mass = []
    rgb_unique_rows = []
    ir_unique_rows = []
    details = []

    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start : start + batch_size]
        query_images = load_batch([pair["query"] for pair in batch_pairs], transform).to(device)
        gallery_images = load_batch([pair["gallery"] for pair in batch_pairs], transform).to(device)

        query_embeddings, query_scores = forward_embeddings_and_scores(model, query_images)
        gallery_embeddings, gallery_scores = forward_embeddings_and_scores(model, gallery_images)

        query_probs = query_scores.softmax(dim=1)
        gallery_probs = gallery_scores.softmax(dim=1)
        query_topk = query_scores.topk(k=min(topk_patches, query_scores.shape[1]), dim=1).indices.cpu().numpy()
        gallery_topk = gallery_scores.topk(k=min(topk_patches, gallery_scores.shape[1]), dim=1).indices.cpu().numpy()
        query_rows = query_topk // grid_w
        gallery_rows = gallery_topk // grid_w
        cosines = (query_embeddings * gallery_embeddings).sum(dim=1).cpu().numpy()

        for idx, pair in enumerate(batch_pairs):
            query_set = set(int(item) for item in query_topk[idx].tolist())
            gallery_set = set(int(item) for item in gallery_topk[idx].tolist())
            overlap = len(query_set & gallery_set) / float(len(query_set | gallery_set)) if (query_set or gallery_set) else 0.0
            pair_cosines.append(float(cosines[idx]))
            overlap_ratios.append(float(overlap))

            query_entropy = float((-(query_probs[idx] * query_probs[idx].clamp_min(1e-12).log()).sum()).item())
            gallery_entropy = float((-(gallery_probs[idx] * gallery_probs[idx].clamp_min(1e-12).log()).sum()).item())
            rgb_entropies.append(gallery_entropy)
            ir_entropies.append(query_entropy)
            rgb_top1_mass.append(float(gallery_probs[idx].max().item()))
            ir_top1_mass.append(float(query_probs[idx].max().item()))
            rgb_unique_rows.append(int(len(set(int(item) for item in gallery_rows[idx].tolist()))))
            ir_unique_rows.append(int(len(set(int(item) for item in query_rows[idx].tolist()))))

            if len(details) < 32:
                details.append(
                    {
                        "pid": int(pair["pid"]),
                        "query_path": pair["query"]["path"],
                        "gallery_path": pair["gallery"]["path"],
                        "cosine": float(cosines[idx]),
                        "topk_overlap_ratio": float(overlap),
                        "query_topk_indices": [int(item) for item in query_topk[idx].tolist()],
                        "gallery_topk_indices": [int(item) for item in gallery_topk[idx].tolist()],
                        "query_topk_rows": [int(item) for item in query_rows[idx].tolist()],
                        "gallery_topk_rows": [int(item) for item in gallery_rows[idx].tolist()],
                    }
                )

    return {
        "grid_size": [int(grid_h), int(grid_w)],
        "num_pairs": len(pairs),
        "matched_pair_cosine": summarize(pair_cosines),
        "topk_overlap_ratio": summarize(overlap_ratios),
        "rgb_patch_entropy": summarize(rgb_entropies),
        "ir_patch_entropy": summarize(ir_entropies),
        "rgb_top1_patch_mass": summarize(rgb_top1_mass),
        "ir_top1_patch_mass": summarize(ir_top1_mass),
        "rgb_unique_rows_in_topk": summarize(rgb_unique_rows),
        "ir_unique_rows_in_topk": summarize(ir_unique_rows),
        "examples": details,
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    config["model"]["image_size"] = list(config["dataset"]["image_size"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = ReIDModel(config["model"], num_classes=config["dataset"]["num_classes"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    pairs = sample_pairs(config["dataset"]["root"], args.mode, args.num_pairs, args.pair_seed)
    metrics = compute_metrics(
        model=model,
        pairs=pairs,
        image_size=tuple(config["dataset"]["image_size"]),
        batch_size=args.batch_size,
        topk_patches=args.topk_patches,
        device=device,
    )
    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "topk_patches": args.topk_patches,
        "pair_seed": args.pair_seed,
        "metrics": metrics,
    }
    dump_json(payload, args.output_json)
    print(payload)


if __name__ == "__main__":
    main()
