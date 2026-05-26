import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.sysumm01.datasets.sysumm01 import SYSUTrainDataset, build_test_records, build_transforms
from project.sysumm01.models.reid_model import ReIDModel
from project.sysumm01.utils.config import dump_json, load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate whether LAST patch scores align with a coarse SYSU foreground proxy")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", default="all", choices=["all", "indoor"])
    parser.add_argument("--subset", default="both", choices=["query", "gallery", "both"])
    parser.add_argument("--max-images", type=int, default=512)
    parser.add_argument("--num-vis", type=int, default=12)
    parser.add_argument("--topk-patches", type=int, default=0, help="0 means using model.topk from config")
    parser.add_argument("--mask-keep-ratio", type=float, default=0.55)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def collect_records(dataset_root, mode, subset, max_images, seed):
    query_records, gallery_records = build_test_records(dataset_root, mode=mode)
    if subset == "query":
        records = list(query_records)
    elif subset == "gallery":
        records = list(gallery_records)
    else:
        records = list(query_records) + list(gallery_records)

    rng = random.Random(seed)
    if max_images > 0 and len(records) > max_images:
        records = rng.sample(records, max_images)
    return records


def load_images(records, transform, image_size):
    height, width = image_size
    tensors = []
    resized_images = []
    for record in records:
        image = Image.open(record["path"]).convert("RGB")
        resized = image.resize((width, height), resample=Image.BICUBIC)
        resized_images.append(np.asarray(resized, dtype=np.float32) / 255.0)
        tensors.append(transform(image))
    return torch.stack(tensors, dim=0), np.stack(resized_images, axis=0)


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def compute_center_prior(height, width):
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    prior = np.exp(-0.5 * ((xs / 0.45) ** 2 + (ys / 0.75) ** 2))
    prior -= prior.min()
    prior /= max(float(prior.max()), 1e-6)
    return prior


def build_foreground_proxy(image, keep_ratio):
    height, width = image.shape[:2]
    margin_y = max(1, height // 10)
    margin_x = max(1, width // 10)

    border_pixels = np.concatenate(
        [
            image[:margin_y].reshape(-1, 3),
            image[-margin_y:].reshape(-1, 3),
            image[:, :margin_x].reshape(-1, 3),
            image[:, -margin_x:].reshape(-1, 3),
        ],
        axis=0,
    )
    border_mean = border_pixels.mean(axis=0, keepdims=True)
    border_std = border_pixels.std(axis=0, keepdims=True) + 1e-3
    color_distance = np.sqrt((((image - border_mean) / border_std) ** 2).sum(axis=2))

    low, high = np.percentile(color_distance, [10.0, 90.0])
    if high <= low:
        normalized_distance = np.zeros_like(color_distance)
    else:
        normalized_distance = np.clip((color_distance - low) / (high - low), 0.0, 1.0)

    combined = 0.75 * normalized_distance + 0.25 * compute_center_prior(height, width)
    threshold = float(np.quantile(combined, max(0.0, 1.0 - keep_ratio)))
    pixel_mask = combined >= threshold
    return combined.astype(np.float32), pixel_mask.astype(bool)


def pixel_mask_to_patch_mask(pixel_mask, grid_h, grid_w):
    height, width = pixel_mask.shape
    patch_h = height // grid_h
    patch_w = width // grid_w
    occupancy = np.zeros((grid_h, grid_w), dtype=np.float32)
    for row in range(grid_h):
        for col in range(grid_w):
            patch = pixel_mask[row * patch_h : (row + 1) * patch_h, col * patch_w : (col + 1) * patch_w]
            occupancy[row, col] = float(patch.mean())
    patch_mask = occupancy >= 0.5
    if not patch_mask.any():
        patch_mask[np.unravel_index(int(np.argmax(occupancy)), occupancy.shape)] = True
    if patch_mask.all():
        patch_mask[np.unravel_index(int(np.argmin(occupancy)), occupancy.shape)] = False
    return occupancy.reshape(-1), patch_mask.reshape(-1)


@torch.no_grad()
def forward_scores(model, images):
    outputs = model(images)
    return outputs["patch_scores"]


def save_example_figure(output_path, image, proxy_score, patch_scores, patch_mask, topk_indices, title, grid_h, grid_w):
    height, width = image.shape[:2]
    patch_h = height // grid_h
    patch_w = width // grid_w
    patch_map = patch_scores.reshape(grid_h, grid_w)

    figure, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(image)
    axes[0].set_title(title)
    axes[0].axis("off")

    axes[1].imshow(image)
    axes[1].imshow(proxy_score, cmap="viridis", alpha=0.45)
    axes[1].set_title("Coarse foreground proxy")
    axes[1].axis("off")

    heatmap = axes[2].imshow(patch_map, cmap="magma", interpolation="nearest")
    axes[2].set_title("Patch score grid")
    axes[2].set_xlabel("Patch X")
    axes[2].set_ylabel("Patch Y")
    figure.colorbar(heatmap, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(image)
    axes[3].set_title("Top-k patches vs proxy")
    axes[3].axis("off")
    for index in topk_indices:
        row = int(index) // grid_w
        col = int(index) % grid_w
        is_foreground = bool(patch_mask[int(index)])
        rectangle = mpatches.Rectangle(
            (col * patch_w, row * patch_h),
            patch_w,
            patch_h,
            linewidth=2,
            edgecolor="#00c853" if is_foreground else "#ff1744",
            facecolor="none",
        )
        axes[3].add_patch(rectangle)

    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def init_group_metrics():
    return {
        "foreground_area_ratio": [],
        "top1_fg_hit": [],
        "topk_fg_hit_ratio": [],
        "softmax_fg_mass": [],
        "fg_bg_score_gap": [],
    }


def finalize_group_metrics(values):
    result = {key: summarize(metric_values) for key, metric_values in values.items()}
    area_mean = result["foreground_area_ratio"]["mean"]
    top1_mean = result["top1_fg_hit"]["mean"]
    topk_mean = result["topk_fg_hit_ratio"]["mean"]
    mass_mean = result["softmax_fg_mass"]["mean"]
    result["top1_vs_area_gap"] = float(top1_mean - area_mean)
    result["topk_vs_area_gap"] = float(topk_mean - area_mean)
    result["mass_vs_area_gap"] = float(mass_mean - area_mean)
    result["top1_lift_over_area"] = float(top1_mean / area_mean) if area_mean > 0 else 0.0
    result["topk_lift_over_area"] = float(topk_mean / area_mean) if area_mean > 0 else 0.0
    result["mass_lift_over_area"] = float(mass_mean / area_mean) if area_mean > 0 else 0.0
    return result


def plot_group_summary(metrics_by_group, output_path):
    groups = [group for group in ("overall", "rgb", "ir") if group in metrics_by_group]
    if not groups:
        return

    metric_names = [
        ("foreground_area_ratio", "FG area"),
        ("top1_fg_hit", "Top-1 hit"),
        ("topk_fg_hit_ratio", "Top-k hit"),
        ("softmax_fg_mass", "FG mass"),
    ]

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    positions = np.arange(len(groups))
    colors = ["#37474f", "#1e88e5", "#e53935"]

    for axis, (metric_name, title) in zip(axes, metric_names):
        values = [metrics_by_group[group][metric_name]["mean"] for group in groups]
        axis.bar(positions, values, color=colors[: len(groups)])
        axis.set_xticks(positions)
        axis.set_xticklabels(groups)
        axis.set_ylim(0.0, 1.0)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)

    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def evaluate(model, records, image_size, batch_size, topk_patches, mask_keep_ratio, output_dir, num_vis, device):
    transform = build_transforms(image_size=image_size, training=False)
    grid_h, grid_w = model.backbone.vit.patch_embed.grid_size
    groups = {"overall": init_group_metrics(), "rgb": init_group_metrics(), "ir": init_group_metrics()}
    examples = []
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        image_tensors, resized_images = load_images(batch_records, transform, image_size)
        scores = forward_scores(model, image_tensors.to(device)).cpu().numpy()

        for index, record in enumerate(batch_records):
            score_vector = scores[index]
            score_probs = torch.softmax(torch.from_numpy(score_vector), dim=0).numpy()
            proxy_score, pixel_mask = build_foreground_proxy(resized_images[index], mask_keep_ratio)
            patch_occupancy, patch_mask = pixel_mask_to_patch_mask(pixel_mask, grid_h, grid_w)
            topk = min(topk_patches, score_vector.shape[0])
            topk_indices = np.argsort(score_vector)[-topk:][::-1]
            top1_index = int(np.argmax(score_vector))

            fg_area_ratio = float(patch_mask.mean())
            top1_hit = float(patch_mask[top1_index])
            topk_hit_ratio = float(patch_mask[topk_indices].mean())
            fg_mass = float(score_probs[patch_mask].sum())
            fg_scores = score_vector[patch_mask]
            bg_scores = score_vector[~patch_mask]
            fg_bg_gap = float(fg_scores.mean() - bg_scores.mean()) if fg_scores.size and bg_scores.size else 0.0

            modality_name = record["modality"]
            for group_name in ("overall", modality_name):
                groups[group_name]["foreground_area_ratio"].append(fg_area_ratio)
                groups[group_name]["top1_fg_hit"].append(top1_hit)
                groups[group_name]["topk_fg_hit_ratio"].append(topk_hit_ratio)
                groups[group_name]["softmax_fg_mass"].append(fg_mass)
                groups[group_name]["fg_bg_score_gap"].append(fg_bg_gap)

            if len(examples) < num_vis:
                file_name = "sample_{:03d}_{}_pid{}_cam{}.png".format(
                    len(examples),
                    modality_name,
                    int(record["pid"]),
                    int(record["camid"]),
                )
                save_example_figure(
                    examples_dir / file_name,
                    resized_images[index],
                    proxy_score,
                    score_vector,
                    patch_mask,
                    topk_indices,
                    "{} pid={} cam={}".format(modality_name.upper(), int(record["pid"]), int(record["camid"])),
                    grid_h,
                    grid_w,
                )
                examples.append(
                    {
                        "path": record["path"],
                        "pid": int(record["pid"]),
                        "camid": int(record["camid"]),
                        "modality": modality_name,
                        "top1_fg_hit": top1_hit,
                        "topk_fg_hit_ratio": topk_hit_ratio,
                        "softmax_fg_mass": fg_mass,
                        "foreground_area_ratio": fg_area_ratio,
                        "topk_indices": [int(item) for item in topk_indices.tolist()],
                        "topk_patch_foreground": [bool(patch_mask[item]) for item in topk_indices.tolist()],
                        "patch_foreground_occupancy": [float(item) for item in patch_occupancy.tolist()],
                        "figure": str((examples_dir / file_name).relative_to(output_dir)),
                    }
                )

    metrics_by_group = {
        group_name: finalize_group_metrics(metric_values)
        for group_name, metric_values in groups.items()
        if metric_values["foreground_area_ratio"]
    }
    plot_group_summary(metrics_by_group, output_dir / "foreground_alignment_summary.png")
    return metrics_by_group, examples


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    config["model"]["image_size"] = list(config["dataset"]["image_size"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_dataset = SYSUTrainDataset(
        root=config["dataset"]["root"],
        image_size=tuple(config["dataset"]["image_size"]),
        use_val=config["dataset"].get("use_val", True),
        train_augment="basic",
    )
    model = ReIDModel(config["model"], num_classes=train_dataset.num_classes)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    records = collect_records(
        dataset_root=config["dataset"]["root"],
        mode=args.mode,
        subset=args.subset,
        max_images=args.max_images,
        seed=args.sample_seed,
    )
    if not records:
        raise ValueError("No records selected for analysis")

    topk_patches = args.topk_patches if args.topk_patches > 0 else int(config["model"].get("topk", 1))
    metrics_by_group, examples = evaluate(
        model=model,
        records=records,
        image_size=tuple(config["dataset"]["image_size"]),
        batch_size=args.batch_size,
        topk_patches=topk_patches,
        mask_keep_ratio=args.mask_keep_ratio,
        output_dir=output_dir,
        num_vis=args.num_vis,
        device=device,
    )

    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "subset": args.subset,
        "max_images": len(records),
        "topk_patches": topk_patches,
        "mask_keep_ratio": args.mask_keep_ratio,
        "metrics": metrics_by_group,
        "examples": examples,
        "notes": [
            "Foreground is a coarse proxy built from border-color contrast plus a center prior.",
            "Interpret hit and mass metrics relative to foreground_area_ratio; random patch selection would match the area ratio in expectation.",
        ],
    }
    dump_json(payload, output_dir / "foreground_alignment_summary.json")
    print(payload)


if __name__ == "__main__":
    main()
