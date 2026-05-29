import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from project.sysumm01.datasets.schp_parts import evaluate_label_mask_quality


def parse_args():
    parser = argparse.ArgumentParser("Build an offline SCHP quality index.")
    parser.add_argument(
        "--mask-root",
        default="/home/cgv841/ybj/TVI-LFM/data/schp_masks/lip",
        help="Root containing source subfolders such as sysumm01/ and vcm/.",
    )
    parser.add_argument(
        "--output",
        default="/home/cgv841/ybj/TVI-LFM/data/schp_masks/lip/schp_quality_index.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=["sysumm01", "vcm"],
        help="Source subfolders to scan under mask-root.",
    )
    return parser.parse_args()


def build_entries(mask_root, sources):
    entries = {}
    totals = {}
    for source_name in sources:
        source_root = Path(mask_root) / source_name
        if not source_root.is_dir():
            raise FileNotFoundError("Source mask folder not found: {}".format(source_root))
        mask_paths = sorted(source_root.rglob("*.png"))
        totals[source_name] = len(mask_paths)
        print("[{}] scanning {} masks".format(source_name, len(mask_paths)), flush=True)
        for index, mask_path in enumerate(mask_paths, start=1):
            if index == 1 or index % 5000 == 0 or index == len(mask_paths):
                print("[{}] {}/{}".format(source_name, index, len(mask_paths)), flush=True)
            rel_stem = str(mask_path.relative_to(source_root).with_suffix("")).replace("\\", "/")
            quality_key = "{}/{}".format(source_name, rel_stem)
            mask = np.asarray(Image.open(mask_path), dtype=np.uint8)
            metrics = evaluate_label_mask_quality(mask)
            entries[quality_key] = {
                "quality_ok": bool(metrics["quality_ok"]),
                "quality_score": float(metrics["quality_score"]),
                "foreground_ratio": float(metrics["foreground_ratio"]),
                "bbox_area_ratio": float(metrics["bbox_area_ratio"]),
                "valid_part_count": int(metrics["valid_part_count"]),
                "upper_valid": bool(metrics["upper_valid"]),
                "lower_valid": bool(metrics["lower_valid"]),
                "head_valid": bool(metrics["head_valid"]),
                "shoes_valid": bool(metrics["shoes_valid"]),
            }
    return entries, totals


def summarize(entries):
    score_values = [item["quality_score"] for item in entries.values()]
    ok_values = [1.0 if item["quality_ok"] else 0.0 for item in entries.values()]
    return {
        "count": len(entries),
        "quality_ok_rate": float(sum(ok_values) / max(len(ok_values), 1)),
        "quality_score_mean": float(sum(score_values) / max(len(score_values), 1)),
        "quality_score_min": float(min(score_values) if score_values else 0.0),
        "quality_score_max": float(max(score_values) if score_values else 0.0),
    }


def main():
    args = parse_args()
    entries, totals = build_entries(args.mask_root, args.sources)
    payload = {
        "metadata": {
            "mask_root": os.path.abspath(args.mask_root),
            "sources": list(args.sources),
            "source_counts": totals,
            "quality_ok_rule": {
                "foreground_ratio": [0.12, 0.88],
                "bbox_area_ratio": [0.20, 0.98],
                "upper_valid": True,
                "lower_valid": True,
                "valid_part_count": 3,
            },
        },
        "summary": summarize(entries),
        "entries": entries,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    print("Saved quality index to {}".format(output_path), flush=True)
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
