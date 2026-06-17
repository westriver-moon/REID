import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.sysumm01.datasets.sysumm01 import build_train_records


LIP_LABELS = {
    0: "Background",
    1: "Hat",
    2: "Hair",
    3: "Glove",
    4: "Sunglasses",
    5: "Upper-clothes",
    6: "Dress",
    7: "Coat",
    8: "Socks",
    9: "Pants",
    10: "Jumpsuits",
    11: "Scarf",
    12: "Skirt",
    13: "Face",
    14: "Left-arm",
    15: "Right-arm",
    16: "Left-leg",
    17: "Right-leg",
    18: "Left-shoe",
    19: "Right-shoe",
}

LIP_PART_GROUPS = {
    "head": [1, 2, 4, 13],
    "upper": [3, 5, 6, 7, 11, 14, 15],
    "lower": [9, 10, 12, 16, 17],
    "shoes": [8, 18, 19],
}

PART_COLORS = {
    "head": (255, 70, 70),
    "upper": (40, 160, 255),
    "lower": (60, 210, 110),
    "shoes": (255, 190, 45),
}


def parse_args():
    parser = argparse.ArgumentParser("Validate SCHP parsing quality on SYSU/VCM samples.")
    parser.add_argument("--output-dir", default="logs/schp_quality/sysu_vcm_ir_lip")
    parser.add_argument("--sysu-root", default="/home/cgv841/datasets/SYSU-MM01")
    parser.add_argument("--vcm-root", default="/home/cgv841/datasets/HITSZ-VCM")
    parser.add_argument(
        "--vcm-index",
        default="/home/cgv841/ybj/TVI-LFM/project/sysumm01/datasets/vcm_index/vcm_train_tracklets.json",
    )
    parser.add_argument("--schp-root", default="/home/cgv841/ybj/TVI-LFM/external/Self-Correction-Human-Parsing")
    parser.add_argument("--checkpoint", default="/home/cgv841/ybj/TVI-LFM/pretrained/schp/exp-schp-201908261155-lip.pth")
    parser.add_argument("--dataset", default="lip", choices=["lip"])
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--num-sysu-ir", type=int, default=80)
    parser.add_argument("--num-vcm-ir", type=int, default=80)
    parser.add_argument("--num-sysu-rgb", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-schp", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-visuals", type=int, default=48)
    parser.add_argument("--grid-height", type=int, default=23)
    parser.add_argument("--grid-width", type=int, default=11)
    return parser.parse_args()


def read_vcm_tracklets(index_path):
    with open(index_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return payload.get("tracklets", [])
    return payload


def choose_records(args):
    rng = random.Random(args.seed)
    records = []

    _, sysu_ir_records, _ = build_train_records(args.sysu_root, use_val=True, train_modality="both")
    sysu_rgb_records, _, _ = build_train_records(args.sysu_root, use_val=True, train_modality="both")
    for source_records, count, subset in (
        (sysu_ir_records, args.num_sysu_ir, "sysu_ir"),
        (sysu_rgb_records, args.num_sysu_rgb, "sysu_rgb"),
    ):
        picked = rng.sample(source_records, min(count, len(source_records))) if count > 0 else []
        for item in picked:
            records.append(
                {
                    "source": "sysu",
                    "subset": subset,
                    "path": item["path"],
                    "pid": int(item["pid"]),
                    "camid": int(item["camid"]),
                    "modality": item["modality"],
                }
            )

    vcm_ir_frames = []
    for tracklet in read_vcm_tracklets(args.vcm_index):
        if str(tracklet.get("modality", "")).lower() != "ir":
            continue
        frames = tracklet.get("frames") or tracklet.get("frame_paths") or []
        if not frames:
            continue
        frame = frames[len(frames) // 2]
        full_path = frame if os.path.isabs(frame) else os.path.join(args.vcm_root, frame)
        vcm_ir_frames.append(
            {
                "source": "vcm",
                "subset": "vcm_ir",
                "path": full_path,
                "pid": int(tracklet["pid"]),
                "camid": int(tracklet["camid"]),
                "modality": "ir",
                "tracklet_id": tracklet.get("tracklet_id"),
            }
        )
    records.extend(rng.sample(vcm_ir_frames, min(args.num_vcm_ir, len(vcm_ir_frames))))
    rng.shuffle(records)
    return records


def safe_link_or_copy(src, dst):
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def stage_inputs(records, input_dir, manifest_path, force=False):
    input_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for path in input_dir.iterdir():
            if path.is_file() or path.is_symlink():
                path.unlink()
    rows = []
    for index, record in enumerate(records):
        src = Path(record["path"])
        if not src.is_file():
            continue
        ext = src.suffix.lower() if src.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp") else ".jpg"
        name = "{:05d}_{}_pid{}_cam{}{}".format(index, record["subset"], record["pid"], record["camid"], ext)
        dst = input_dir / name
        safe_link_or_copy(str(src), dst)
        row = dict(record)
        row["input_name"] = name
        row["input_path"] = str(dst)
        row["mask_name"] = Path(name).with_suffix(".png").name
        rows.append(row)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
    return rows


def run_schp(args, input_dir, mask_dir):
    mask_dir.mkdir(parents=True, exist_ok=True)
    input_dir = input_dir.resolve()
    mask_dir = mask_dir.resolve()
    command = [
        sys.executable,
        "simple_extractor.py",
        "--dataset",
        args.dataset,
        "--model-restore",
        args.checkpoint,
        "--gpu",
        args.gpu,
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(mask_dir),
    ]
    with open(Path(args.output_dir) / "schp_command.txt", "w", encoding="utf-8") as handle:
        handle.write(" ".join(command) + "\n")
    subprocess.run(command, cwd=args.schp_root, check=True)


def mask_for_labels(mask, labels):
    result = np.zeros(mask.shape, dtype=bool)
    for label in labels:
        result |= mask == label
    return result


def bbox_from_mask(binary):
    ys, xs = np.where(binary)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def patch_occupancy(binary, grid_h, grid_w):
    h, w = binary.shape
    values = np.zeros((grid_h, grid_w), dtype=np.float32)
    for r in range(grid_h):
        y0 = int(round(r * h / float(grid_h)))
        y1 = int(round((r + 1) * h / float(grid_h)))
        for c in range(grid_w):
            x0 = int(round(c * w / float(grid_w)))
            x1 = int(round((c + 1) * w / float(grid_w)))
            patch = binary[y0:y1, x0:x1]
            values[r, c] = float(patch.mean()) if patch.size else 0.0
    return values


def colorize_parts(mask):
    overlay = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for name, labels in LIP_PART_GROUPS.items():
        overlay[mask_for_labels(mask, labels)] = PART_COLORS[name]
    return overlay


def evaluate_mask(mask):
    h, w = mask.shape
    image_area = float(h * w)
    foreground = mask > 0
    foreground_pixels = int(foreground.sum())
    foreground_ratio = foreground_pixels / image_area
    bbox = bbox_from_mask(foreground)
    bbox_area_ratio = 0.0
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        bbox_area_ratio = ((x1 - x0) * (y1 - y0)) / image_area

    metrics = {
        "foreground_ratio": foreground_ratio,
        "bbox_area_ratio": bbox_area_ratio,
        "bbox": bbox,
    }
    valid_parts = 0
    for name, labels in LIP_PART_GROUPS.items():
        part = mask_for_labels(mask, labels)
        ratio = float(part.sum()) / image_area
        fg_ratio = float(part.sum()) / max(float(foreground_pixels), 1.0)
        patch_occ = patch_occupancy(part, 23, 11)
        patch_count = int((patch_occ >= 0.25).sum())
        is_valid = ratio >= 0.005 and patch_count >= 1
        valid_parts += int(is_valid)
        metrics[name + "_ratio"] = ratio
        metrics[name + "_fg_ratio"] = fg_ratio
        metrics[name + "_patch_count"] = patch_count
        metrics[name + "_valid"] = bool(is_valid)
    metrics["valid_part_count"] = valid_parts
    metrics["quality_ok"] = bool(
        0.12 <= foreground_ratio <= 0.88
        and 0.20 <= bbox_area_ratio <= 0.98
        and metrics["upper_valid"]
        and metrics["valid_part_count"] >= 3
    )
    return metrics


def save_visual(record, mask, metrics, out_path, grid_h, grid_w):
    image = Image.open(record["input_path"]).convert("RGB")
    image_np = np.asarray(image.resize((144, 288), Image.BICUBIC), dtype=np.float32) / 255.0
    mask_resized = np.asarray(Image.fromarray(mask).resize((144, 288), Image.NEAREST))
    part_overlay = colorize_parts(mask_resized)
    foreground = mask_resized > 0
    fg_grid = patch_occupancy(foreground, grid_h, grid_w)
    part_grid = patch_occupancy(mask_resized > 0, grid_h, grid_w)

    figure, axes = plt.subplots(1, 4, figsize=(16, 5))
    axes[0].imshow(image_np)
    axes[0].set_title("{} pid={} cam={}".format(record["subset"], record["pid"], record["camid"]))
    axes[0].axis("off")

    axes[1].imshow(image_np)
    axes[1].imshow(part_overlay, alpha=0.48)
    axes[1].set_title("SCHP grouped parts")
    axes[1].axis("off")

    axes[2].imshow(foreground, cmap="gray")
    axes[2].set_title("foreground={:.2f}, parts={}".format(metrics["foreground_ratio"], metrics["valid_part_count"]))
    axes[2].axis("off")

    axes[3].imshow(image_np)
    axes[3].imshow(part_grid, cmap="magma", alpha=0.55, interpolation="nearest", extent=(0, 144, 288, 0))
    axes[3].set_title("patch prior grid {}x{}".format(grid_h, grid_w))
    axes[3].axis("off")

    figure.tight_layout()
    figure.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}


def analyze_outputs(args, rows, mask_dir, visual_dir):
    visual_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for row in rows:
        mask_path = mask_dir / row["mask_name"]
        if not mask_path.is_file():
            continue
        mask = np.asarray(Image.open(mask_path), dtype=np.uint8)
        metrics = evaluate_mask(mask)
        result = dict(row)
        result.update(metrics)
        results.append(result)

    results.sort(key=lambda item: (item["quality_ok"], item["valid_part_count"], item["foreground_ratio"]))
    selected = results[: max(0, args.max_visuals // 2)] + results[-max(0, args.max_visuals - args.max_visuals // 2) :]
    selected_names = set()
    for item in selected:
        if item["input_name"] in selected_names:
            continue
        selected_names.add(item["input_name"])
        mask = np.asarray(Image.open(mask_dir / item["mask_name"]), dtype=np.uint8)
        figure_name = Path(item["input_name"]).with_suffix(".png").name
        item["figure"] = "visuals/" + figure_name
        save_visual(item, mask, item, visual_dir / figure_name, args.grid_height, args.grid_width)

    metrics_by_subset = {}
    for subset in sorted({item["subset"] for item in results}):
        group = [item for item in results if item["subset"] == subset]
        metrics_by_subset[subset] = {
            "count": len(group),
            "quality_ok_rate": sum(item["quality_ok"] for item in group) / float(max(len(group), 1)),
            "foreground_ratio": summarize([item["foreground_ratio"] for item in group]),
            "bbox_area_ratio": summarize([item["bbox_area_ratio"] for item in group]),
            "valid_part_count": summarize([item["valid_part_count"] for item in group]),
        }
        for part in LIP_PART_GROUPS:
            metrics_by_subset[subset][part + "_valid_rate"] = sum(item[part + "_valid"] for item in group) / float(max(len(group), 1))
            metrics_by_subset[subset][part + "_ratio"] = summarize([item[part + "_ratio"] for item in group])

    return results, metrics_by_subset


def write_reports(output_dir, results, metrics_by_subset):
    csv_path = output_dir / "schp_quality_samples.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    payload = {
        "metrics_by_subset": metrics_by_subset,
        "labels": LIP_LABELS,
        "part_groups": LIP_PART_GROUPS,
        "notes": [
            "No ground-truth parsing labels are available here; quality_ok is a heuristic diagnostic.",
            "A sample is quality_ok when foreground and bbox coverage are plausible, upper body exists, and at least 3 grouped parts are present.",
            "Use the visual report for the final decision, especially for IR images.",
        ],
    }
    with open(output_dir / "schp_quality_summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    cards = []
    for item in results:
        if "figure" not in item:
            continue
        cards.append(
            '<section><img src="{fig}"><p>{name} | {subset} | ok={ok} | fg={fg:.3f} | parts={parts}</p></section>'.format(
                fig=item["figure"],
                name=item["input_name"],
                subset=item["subset"],
                ok=item["quality_ok"],
                fg=item["foreground_ratio"],
                parts=item["valid_part_count"],
            )
        )
    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>SCHP Parsing Quality</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f8f8f8; color: #222; }}
    pre {{ background: white; padding: 12px; border: 1px solid #ddd; overflow-x: auto; }}
    section {{ background: white; padding: 12px; border: 1px solid #ddd; margin-bottom: 16px; }}
    img {{ width: 100%; max-width: 1400px; display: block; }}
  </style>
</head>
<body>
  <h1>SCHP Parsing Quality</h1>
  <pre>{summary}</pre>
  {cards}
</body>
</html>
""".format(summary=json.dumps(metrics_by_subset, indent=2, sort_keys=True), cards="\n".join(cards))
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    input_dir = output_dir / "inputs"
    mask_dir = output_dir / "masks"
    visual_dir = output_dir / "visuals"
    manifest_path = output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.force or not manifest_path.is_file():
        records = choose_records(args)
        rows = stage_inputs(records, input_dir, manifest_path, force=args.force)
    else:
        rows = json.loads(manifest_path.read_text())

    if args.run_schp:
        if not Path(args.checkpoint).is_file():
            raise FileNotFoundError("SCHP checkpoint not found: {}".format(args.checkpoint))
        run_schp(args, input_dir, mask_dir)

    results, metrics_by_subset = analyze_outputs(args, rows, mask_dir, visual_dir)
    write_reports(output_dir, results, metrics_by_subset)
    print("Samples staged:", len(rows))
    print("Masks analyzed:", len(results))
    print("Report:", output_dir / "index.html")
    print(json.dumps(metrics_by_subset, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
