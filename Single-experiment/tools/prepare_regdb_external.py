#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def read_split_file(path, modality, camid):
    samples = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError("Invalid RegDB split line {} in {}: {}".format(line_number, path, line))
            rel_path, pid = parts[0], int(parts[1])
            samples.append(
                {
                    "path": rel_path,
                    "pid": pid,
                    "camid": camid,
                    "modality": modality,
                }
            )
    return samples


def main():
    parser = argparse.ArgumentParser(description="Build an ExternalRGBIRDataset index for RegDB")
    parser.add_argument("--root", default="/home/cgv841/datasets/RegDB")
    parser.add_argument("--output", required=True)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    idx_dir = root / "idx"
    visible_file = idx_dir / "{}_visible_{}.txt".format(args.split, args.trial)
    thermal_file = idx_dir / "{}_thermal_{}.txt".format(args.split, args.trial)
    for path in (root, idx_dir, visible_file, thermal_file):
        if not path.exists():
            raise FileNotFoundError(path)

    samples = []
    samples.extend(read_split_file(visible_file, modality="rgb", camid=1))
    samples.extend(read_split_file(thermal_file, modality="ir", camid=2))

    missing = [sample["path"] for sample in samples if not (root / sample["path"]).is_file()]
    if missing:
        raise FileNotFoundError("Missing RegDB images, first examples: {}".format(missing[:10]))

    pids = sorted({sample["pid"] for sample in samples})
    payload = {
        "metadata": {
            "dataset": "RegDB",
            "root": str(root),
            "split": args.split,
            "trial": args.trial,
            "num_pids": len(pids),
            "num_samples": len(samples),
            "num_rgb": sum(1 for sample in samples if sample["modality"] == "rgb"),
            "num_ir": sum(1 for sample in samples if sample["modality"] == "ir"),
            "note": "Prepared from RegDB idx split files for ExternalRGBIRDataset.",
        },
        "samples": samples,
    }

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print("Wrote {}".format(output))
    print("pids={} samples={} rgb={} ir={}".format(
        payload["metadata"]["num_pids"],
        payload["metadata"]["num_samples"],
        payload["metadata"]["num_rgb"],
        payload["metadata"]["num_ir"],
    ))


if __name__ == "__main__":
    main()
