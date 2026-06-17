#!/usr/bin/env python
import argparse
import collections
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.sysumm01.datasets.schp_parts import (  # noqa: E402
    evaluate_label_mask_quality,
    make_quality_key_from_relative_path,
)
from prepare_vcm_safe import (  # noqa: E402
    VCM_FRAME_RE,
    make_relative_to_root,
    modality_from_code,
    read_lines,
)


def parse_args():
    parser = argparse.ArgumentParser(
        "Score every HITSZ-VCM IR frame with SCHP and build a filtered tracklet index."
    )
    parser.add_argument("--mode", choices=("score", "finalize"), required=True)
    parser.add_argument("--root", default="/home/cgv841/datasets/HITSZ-VCM")
    parser.add_argument(
        "--base-quality-index",
        default="/home/cgv841/ybj/TVI-LFM/data/schp_masks/lip/schp_quality_index.json",
    )
    parser.add_argument(
        "--shard-output",
        default="/home/cgv841/ybj/TVI-LFM/data/schp_quality/vcm_ir_quality_shard_{shard_index}.json",
    )
    parser.add_argument(
        "--shard-glob",
        default="/home/cgv841/ybj/TVI-LFM/data/schp_quality/vcm_ir_quality_shard_*.json",
    )
    parser.add_argument(
        "--quality-output",
        default="/home/cgv841/ybj/TVI-LFM/data/schp_quality/vcm_ir_quality_full.json",
    )
    parser.add_argument(
        "--tracklet-output",
        default="/home/cgv841/ybj/TVI-LFM/project/sysumm01/datasets/vcm_index/vcm_train_tracklets_schp_filtered.json",
    )
    parser.add_argument(
        "--frames-output",
        default="/home/cgv841/ybj/TVI-LFM/project/sysumm01/datasets/vcm_index/vcm_train_frames_schp_filtered.jsonl",
    )
    parser.add_argument(
        "--meta-output",
        default="/home/cgv841/ybj/TVI-LFM/project/sysumm01/datasets/vcm_index/vcm_ir_schp_filtered_meta.json",
    )
    parser.add_argument("--schp-root", default="/home/cgv841/ybj/TVI-LFM/external/Self-Correction-Human-Parsing")
    parser.add_argument("--checkpoint", default="/home/cgv841/ybj/TVI-LFM/pretrained/schp/exp-schp-201908261155-lip.pth")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-frames-per-tracklet", type=int, default=24)
    return parser.parse_args()


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
    os.replace(str(temp_path), str(path))


def load_entries(path, source_name=None):
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload.get("entries", payload)
    if source_name is None:
        return dict(entries)
    prefix = source_name + "/"
    return {key: value for key, value in entries.items() if key.startswith(prefix)}


def quality_record(metrics):
    return {
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


def summarize(entries):
    values = list(entries.values())
    scores = [float(item["quality_score"]) for item in values]
    ok_count = sum(bool(item["quality_ok"]) for item in values)
    return {
        "count": len(values),
        "quality_ok_count": ok_count,
        "quality_ok_rate": float(ok_count / max(len(values), 1)),
        "quality_score_mean": float(sum(scores) / max(len(scores), 1)),
        "quality_score_min": float(min(scores) if scores else 0.0),
        "quality_score_max": float(max(scores) if scores else 0.0),
    }


def collect_full_ir_tracklets(root):
    root = Path(root).resolve()
    frame_names = read_lines(root / "train_name.txt")
    grouped = collections.defaultdict(list)
    for frame_name in frame_names:
        match = VCM_FRAME_RE.match(Path(frame_name).name)
        if match is None:
            raise ValueError("Unable to parse VCM frame name: {}".format(frame_name))
        modality = modality_from_code(match.group("modality"))
        if modality != "ir":
            continue
        key = (int(match.group("pid")), int(match.group("camera")), int(match.group("track")))
        grouped[key].append((int(match.group("frame")), make_relative_to_root(root, frame_name)))

    pids = {key[0] for key in grouped}
    pid_to_label = {pid: index for index, pid in enumerate(sorted(pids))}
    tracklets = []
    for tracklet_id, ((pid, camid, source_track), indexed_frames) in enumerate(sorted(grouped.items())):
        frames = [frame for _, frame in sorted(indexed_frames)]
        missing = [frame for frame in frames if not (root / frame).is_file()]
        if missing:
            raise FileNotFoundError("Missing VCM IR frame: {}".format(root / missing[0]))
        tracklets.append(
            {
                "tracklet_id": int(tracklet_id),
                "pid": int(pid),
                "label": int(pid_to_label[pid]),
                "camid": int(camid),
                "modality": "ir",
                "modality_code": 1,
                "frames": frames,
                "num_frames_original": len(frames),
                "source_track": int(source_track),
            }
        )
    return tracklets


def unique_frames(tracklets):
    seen = set()
    frames = []
    for tracklet in tracklets:
        for frame in tracklet["frames"]:
            if frame in seen:
                continue
            seen.add(frame)
            frames.append(frame)
    return frames


def stage_chunk(root, frames, stage_dir):
    input_dir = stage_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, frame in enumerate(frames):
        source = root / frame
        staged_name = "{:07d}{}".format(index, source.suffix.lower() or ".jpg")
        os.symlink(str(source), str(input_dir / staged_name))
        manifest.append((frame, Path(staged_name).with_suffix(".png").name))
    return input_dir, manifest


def run_schp(args, input_dir, output_dir):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    command = [
        sys.executable,
        "simple_extractor.py",
        "--dataset",
        "lip",
        "--model-restore",
        args.checkpoint,
        "--gpu",
        "0",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=args.schp_root, check=True, env=env)


def score_mode(args, tracklets):
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards)")
    root = Path(args.root).resolve()
    output = args.shard_output.format(shard_index=args.shard_index)
    base_entries = load_entries(args.base_quality_index, source_name="vcm")
    shard_entries = load_entries(output, source_name="vcm")
    frames = unique_frames(tracklets)
    all_missing = [
        frame
        for frame in frames
        if make_quality_key_from_relative_path(frame, "vcm") not in base_entries
    ]
    assigned_frames = all_missing[args.shard_index :: args.num_shards]
    shard_frames = [
        frame
        for frame in assigned_frames
        if make_quality_key_from_relative_path(frame, "vcm") not in shard_entries
    ]
    print("VCM IR frames: {}".format(len(frames)), flush=True)
    print("Reusable scores: {}".format(len(base_entries)), flush=True)
    print(
        "Shard {}/{} assigned: {}, remaining to score: {}".format(
            args.shard_index, args.num_shards, len(assigned_frames), len(shard_frames)
        ),
        flush=True,
    )
    for start in range(0, len(shard_frames), args.chunk_size):
        chunk = shard_frames[start : start + args.chunk_size]
        with tempfile.TemporaryDirectory(prefix="vcm_ir_schp_") as tmp:
            temp_dir = Path(tmp)
            input_dir, manifest = stage_chunk(root, chunk, temp_dir)
            output_dir = temp_dir / "masks"
            run_schp(args, input_dir, output_dir)
            for frame, mask_name in manifest:
                mask = np.asarray(Image.open(output_dir / mask_name), dtype=np.uint8)
                shard_entries[make_quality_key_from_relative_path(frame, "vcm")] = quality_record(
                    evaluate_label_mask_quality(mask)
                )
        payload = {
            "metadata": {
                "dataset": "HITSZ-VCM",
                "modality": "ir",
                "kind": "schp_quality_shard",
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            },
            "summary": summarize(shard_entries),
            "entries": shard_entries,
        }
        atomic_write_json(output, payload)
        print("Shard {}/{} processed {}/{}".format(
            args.shard_index, args.num_shards, min(start + args.chunk_size, len(shard_frames)), len(shard_frames)
        ), flush=True)
    print("Saved shard scores to {}".format(output), flush=True)


def filtered_frames(tracklet, entries, max_frames):
    scored = []
    for frame in tracklet["frames"]:
        key = make_quality_key_from_relative_path(frame, "vcm")
        entry = entries[key]
        scored.append((frame, entry))
    quality_ok = [item for item in scored if bool(item[1]["quality_ok"])]
    candidates = quality_ok if quality_ok else scored
    candidates.sort(key=lambda item: float(item[1]["quality_score"]), reverse=True)
    return candidates[:max_frames], len(quality_ok)


def finalize_mode(args, tracklets):
    base_entries = load_entries(args.base_quality_index, source_name="vcm")
    entries = dict(base_entries)
    shard_paths = sorted(glob.glob(args.shard_glob))
    if not shard_paths:
        raise FileNotFoundError("No shard files matched {}".format(args.shard_glob))
    for shard_path in shard_paths:
        entries.update(load_entries(shard_path, source_name="vcm"))
    frames = unique_frames(tracklets)
    expected_keys = {make_quality_key_from_relative_path(frame, "vcm") for frame in frames}
    missing = sorted(expected_keys - set(entries))
    if missing:
        raise RuntimeError("Missing {} SCHP scores, first: {}".format(len(missing), missing[0]))
    entries = {key: entries[key] for key in sorted(expected_keys)}

    output_tracklets = []
    output_frames = []
    fallback_tracklets = 0
    original_count = 0
    filtered_count = 0
    quality_ok_count = 0
    for tracklet in tracklets:
        selected, num_quality_ok = filtered_frames(tracklet, entries, args.max_frames_per_tracklet)
        fallback_tracklets += int(num_quality_ok == 0)
        original_count += len(tracklet["frames"])
        filtered_count += len(selected)
        quality_ok_count += num_quality_ok
        output = dict(tracklet)
        output["frames"] = [frame for frame, _ in selected]
        output["frame_quality_scores"] = [float(entry["quality_score"]) for _, entry in selected]
        output["num_frames"] = len(selected)
        output["num_quality_ok_frames_original"] = int(num_quality_ok)
        output["selection_strategy"] = "quality_ok_then_score_top{}".format(args.max_frames_per_tracklet)
        output_tracklets.append(output)
        for frame_index, (frame, entry) in enumerate(selected):
            output_frames.append(
                {
                    "tracklet_id": output["tracklet_id"],
                    "frame_index": frame_index,
                    "pid": output["pid"],
                    "label": output["label"],
                    "camid": output["camid"],
                    "modality": "ir",
                    "path": frame,
                    "quality_ok": bool(entry["quality_ok"]),
                    "quality_score": float(entry["quality_score"]),
                }
            )

    quality_payload = {
        "metadata": {
            "dataset": "HITSZ-VCM",
            "modality": "ir",
            "kind": "schp_quality_full",
            "root": str(Path(args.root).resolve()),
            "reused_base_entries": len(base_entries),
            "merged_shards": shard_paths,
        },
        "summary": summarize(entries),
        "entries": entries,
    }
    tracklet_payload = {
        "metadata": {
            "dataset": "HITSZ-VCM",
            "split": "train",
            "modality": "ir",
            "root": str(Path(args.root).resolve()),
            "quality_index": str(Path(args.quality_output).resolve()),
            "selection_strategy": "quality_ok_then_score_top{}".format(args.max_frames_per_tracklet),
            "max_frames_per_tracklet": args.max_frames_per_tracklet,
        },
        "tracklets": output_tracklets,
    }
    meta = {
        "dataset": "HITSZ-VCM",
        "split": "train",
        "modality": "ir",
        "num_tracklets": len(output_tracklets),
        "num_frames_original": original_count,
        "num_frames_filtered": filtered_count,
        "num_quality_ok_frames_original": quality_ok_count,
        "fallback_tracklets": fallback_tracklets,
        "max_frames_per_tracklet": args.max_frames_per_tracklet,
        "selection_strategy": "quality_ok_then_score_top{}".format(args.max_frames_per_tracklet),
        "quality_summary": quality_payload["summary"],
    }
    atomic_write_json(args.quality_output, quality_payload)
    atomic_write_json(args.tracklet_output, tracklet_payload)
    Path(args.frames_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.frames_output, "w", encoding="utf-8") as handle:
        for item in output_frames:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    atomic_write_json(args.meta_output, meta)
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)
    print("Saved full quality index to {}".format(args.quality_output), flush=True)
    print("Saved filtered tracklets to {}".format(args.tracklet_output), flush=True)


def main():
    args = parse_args()
    tracklets = collect_full_ir_tracklets(args.root)
    print("Recovered {} full VCM IR tracklets".format(len(tracklets)), flush=True)
    if args.mode == "score":
        score_mode(args, tracklets)
    else:
        finalize_mode(args, tracklets)


if __name__ == "__main__":
    main()
