#!/usr/bin/env python
import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


LLCM_NAME_RE = re.compile(
    r"^(?P<pid>\d+)_c(?P<camid>\d+)_s(?P<scene>\d+)_f(?P<frame>\d+)_(?P<suffix>vis|nir)\.[^.]+$",
    re.IGNORECASE,
)


def read_split(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError("{}:{} must contain '<path> <label>'".format(path, line_no))
            records.append((parts[0].replace("\\", "/"), int(parts[1])))
    return records


def parse_image_name(rel_path):
    match = LLCM_NAME_RE.match(Path(rel_path).name)
    if not match:
        raise ValueError("Unable to parse LLCM filename: {}".format(rel_path))
    suffix = match.group("suffix").lower()
    return {
        "pid": int(match.group("pid")),
        "camid": int(match.group("camid")),
        "scene": int(match.group("scene")),
        "frame": int(match.group("frame")),
        "modality": "rgb" if suffix == "vis" else "ir",
    }


def build_index(root, split):
    root = Path(root).resolve()
    idx_dir = root / "idx"
    split_files = {
        "train": [
            ("rgb", idx_dir / "train_vis.txt"),
            ("ir", idx_dir / "train_nir.txt"),
        ],
    }
    if split not in split_files:
        raise ValueError("Unsupported LLCM split: {}".format(split))

    raw_samples = []
    missing = []
    for expected_modality, split_file in split_files[split]:
        if not split_file.is_file():
            raise FileNotFoundError("Missing LLCM split file: {}".format(split_file))
        for rel_path, split_label in read_split(split_file):
            info = parse_image_name(rel_path)
            if info["modality"] != expected_modality:
                raise ValueError(
                    "Modality mismatch in {}: expected {}, parsed {}".format(
                        rel_path,
                        expected_modality,
                        info["modality"],
                    )
                )
            abs_path = root / rel_path
            if not abs_path.is_file():
                missing.append(rel_path)
                continue
            raw_samples.append(
                {
                    "source": "llcm",
                    "sample_type": "image",
                    "pid": info["pid"],
                    "split_label": split_label,
                    "camid": info["camid"],
                    "scene": info["scene"],
                    "frame": info["frame"],
                    "modality": info["modality"],
                    "path": rel_path,
                }
            )

    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            "Missing {} LLCM image files under {}. First missing: {}".format(
                len(missing),
                root,
                preview,
            )
        )
    if not raw_samples:
        raise RuntimeError("No LLCM {} samples found under {}".format(split, root))

    pids = sorted({sample["pid"] for sample in raw_samples})
    pid_to_label = {pid: label for label, pid in enumerate(pids)}
    for sample in raw_samples:
        sample["label"] = pid_to_label[sample["pid"]]

    modality_counts = Counter(sample["modality"] for sample in raw_samples)
    cam_counts = Counter(sample["camid"] for sample in raw_samples)
    per_pid_modalities = defaultdict(set)
    for sample in raw_samples:
        per_pid_modalities[sample["pid"]].add(sample["modality"])
    paired_pids = sum(1 for modalities in per_pid_modalities.values() if {"rgb", "ir"} <= modalities)

    return {
        "metadata": {
            "name": "llcm",
            "dataset": "LLCM",
            "root": str(root),
            "split": split,
            "num_samples": len(raw_samples),
            "num_pids": len(pids),
            "num_rgb": int(modality_counts.get("rgb", 0)),
            "num_ir": int(modality_counts.get("ir", 0)),
            "num_cameras": len(cam_counts),
            "paired_train_pids": paired_pids,
            "idx_files": [str(path.resolve()) for _, path in split_files[split]],
        },
        "samples": raw_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare LLCM train image index for external RGB-IR pretraining")
    parser.add_argument("--root", required=True, help="LLCM root directory")
    parser.add_argument("--split", default="train", choices=["train"])
    parser.add_argument(
        "--output",
        default=None,
        help="Output json path; defaults to TVI-LFM/data/external_indices/llcm_train_images.json",
    )
    args = parser.parse_args()

    payload = build_index(args.root, args.split)
    default_output = Path(__file__).resolve().parents[1] / "data" / "external_indices" / "llcm_train_images.json"
    output = Path(args.output).resolve() if args.output else default_output
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    meta = payload["metadata"]
    print(
        "Wrote LLCM {split} index: {samples} images, {pids} pids, {rgb} rgb, {ir} ir, {paired} paired pids -> {output}".format(
            split=meta["split"],
            samples=meta["num_samples"],
            pids=meta["num_pids"],
            rgb=meta["num_rgb"],
            ir=meta["num_ir"],
            paired=meta["paired_train_pids"],
            output=output,
        )
    )


if __name__ == "__main__":
    main()
