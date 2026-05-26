#!/usr/bin/env python
import argparse
import json
import os
import random
import re
import shutil
from pathlib import Path


ACADEMIC_NOTICE = (
    "HITSZ-VCM is for academic use only. Do not redistribute the dataset. "
    "This script only creates local indexes and never packages or uploads data."
)

VCM_FRAME_RE = re.compile(
    r"^(?P<pid>\d+)M(?P<modality>\d+)D(?P<camera>\d+)T(?P<track>\d+)F(?P<frame>\d+)(?P<ext>\.[^.]+)$",
    re.IGNORECASE,
)


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in ("1", "true", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value, got {}".format(value))


def read_lines(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def parse_track_info_line(line, line_no, name):
    values = []
    for token in line.replace(",", " ").split():
        try:
            values.append(int(token))
        except ValueError:
            continue
    if len(values) < 4:
        raise ValueError("{} line {} must contain at least 4 integers: {}".format(name, line_no, line))
    # Official HITSZ-VCM info files use:
    #   modality start_frame end_frame pid camid
    # Older/local indexes may use:
    #   start_frame end_frame pid camid [extra...]
    if len(values) >= 5 and values[0] in (1, 2) and values[1] <= values[2]:
        modality_code, start, end, pid, camid = values[:5]
        return start, end, pid, camid, values[5:], modality_code
    return values[0], values[1], values[2], values[3], values[4:], None


def modality_from_code(code):
    if int(code) == 1:
        return "ir"
    if int(code) == 2:
        return "rgb"
    raise ValueError("Unsupported VCM modality code: {}".format(code))


def free_bytes(path):
    usage = shutil.disk_usage(str(path))
    return usage.free


def require_min_free(path, min_free_gb, context):
    free = free_bytes(path)
    required = int(float(min_free_gb) * 1024 ** 3)
    if free < required:
        raise RuntimeError(
            "{} requires at least {:.1f}GB free under {}, but only {:.1f}GB is available".format(
                context,
                float(min_free_gb),
                path,
                free / 1024 ** 3,
            )
        )


def select_frames(frames, max_frames, sampling, seed, tracklet_id):
    if sampling == "all" or max_frames is None or max_frames <= 0 or len(frames) <= max_frames:
        return list(frames)
    if sampling == "uniform":
        if max_frames == 1:
            return [frames[len(frames) // 2]]
        positions = [
            round(index * (len(frames) - 1) / float(max_frames - 1))
            for index in range(max_frames)
        ]
        return [frames[int(pos)] for pos in positions]
    if sampling == "random":
        rng = random.Random(int(seed) + int(tracklet_id))
        indices = sorted(rng.sample(range(len(frames)), max_frames))
        return [frames[index] for index in indices]
    raise ValueError("Unsupported sampling: {}".format(sampling))


def infer_modality(frame_path):
    parts = Path(frame_path.replace("\\", "/")).parts
    lowered = [part.lower() for part in parts]
    if "rgb" in lowered:
        return "rgb"
    if "ir" in lowered:
        return "ir"
    match = VCM_FRAME_RE.match(Path(frame_path).name)
    if match:
        return modality_from_code(match.group("modality"))
    raise ValueError("Unable to infer modality from frame path: {}".format(frame_path))


def make_relative_to_root(root, path_text):
    path_text = path_text.replace("\\", "/")
    if os.path.isabs(path_text):
        return os.path.relpath(path_text, root).replace("\\", "/")
    if (root / path_text).is_file():
        return path_text
    if (root / "data" / path_text).is_file():
        return ("data/" + path_text).replace("//", "/")
    match = VCM_FRAME_RE.match(Path(path_text).name)
    if match:
        modality = modality_from_code(match.group("modality"))
        ext = match.group("ext").lower()
        return "data/{pid}/{modality}/D{camera}/{frame}{ext}".format(
            pid=match.group("pid"),
            modality=modality,
            camera=match.group("camera"),
            frame=match.group("frame"),
            ext=ext,
        )
    return path_text


def resolve_existing_frame(root, frame_rel):
    candidates = [
        root / frame_rel,
        root / "data" / frame_rel,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def build_split(root, split, name_file, info_file, max_frames, sampling, seed):
    frame_names = [make_relative_to_root(root, item) for item in read_lines(name_file)]
    info_lines = read_lines(info_file)

    parsed = []
    starts = []
    ends = []
    for line_no, line in enumerate(info_lines, start=1):
        start, end, pid, camid, extra, modality_code = parse_track_info_line(line, line_no, info_file.name)
        parsed.append((start, end, pid, camid, extra, modality_code))
        starts.append(start)
        ends.append(end)
    if not parsed:
        raise RuntimeError("No tracklets found in {}".format(info_file))

    one_based = min(starts) >= 1 and max(ends) <= len(frame_names)
    pids = sorted({pid for _, _, pid, _, _, _ in parsed})
    pid_to_label = {pid: index for index, pid in enumerate(pids)}
    tracklets = []
    frames_jsonl = []
    original_frame_count = 0
    limited_frame_count = 0

    for tracklet_id, (start, end, pid, camid, extra, modality_code) in enumerate(parsed):
        begin = start - 1 if one_based else start
        finish = end if one_based else end + 1
        if begin < 0 or finish > len(frame_names) or begin >= finish:
            raise ValueError(
                "Invalid frame range for {} tracklet {}: start={} end={} frame_count={}".format(
                    split,
                    tracklet_id,
                    start,
                    end,
                    len(frame_names),
                )
            )
        original_frames = frame_names[begin:finish]
        selected_frames = select_frames(
            original_frames,
            max_frames=max_frames,
            sampling=sampling,
            seed=seed,
            tracklet_id=tracklet_id,
        )
        modality = modality_from_code(modality_code) if modality_code is not None else infer_modality(selected_frames[0])
        original_frame_count += len(original_frames)
        limited_frame_count += len(selected_frames)
        tracklet = {
            "tracklet_id": tracklet_id,
            "pid": int(pid),
            "label": int(pid_to_label[pid]),
            "camid": int(camid),
            "modality": modality,
            "frames": selected_frames,
            "num_frames": len(selected_frames),
            "num_frames_original": len(original_frames),
            "source_range": [int(start), int(end)],
            "extra": extra,
            "modality_code": None if modality_code is None else int(modality_code),
        }
        tracklets.append(tracklet)
        for frame_idx, frame_rel in enumerate(selected_frames):
            frames_jsonl.append(
                {
                    "split": split,
                    "tracklet_id": tracklet_id,
                    "frame_index": frame_idx,
                    "pid": int(pid),
                    "label": int(pid_to_label[pid]),
                    "camid": int(camid),
                    "modality": modality,
                    "path": frame_rel,
                }
            )

    return {
        "metadata": {
            "dataset": "HITSZ-VCM",
            "split": split,
            "root": str(root),
            "name_file": str(name_file),
            "track_info_file": str(info_file),
            "index_base": 1 if one_based else 0,
            "num_frames_in_name_file": len(frame_names),
        },
        "tracklets": tracklets,
        "frames": frames_jsonl,
        "num_pids": len(pids),
        "num_tracklets": len(tracklets),
        "num_frames_original": original_frame_count,
        "num_frames_limited": limited_frame_count,
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def estimate_copy_bytes(root, frames):
    total = 0
    missing = []
    for frame_rel in frames:
        source = resolve_existing_frame(root, frame_rel)
        if not source.is_file():
            missing.append(str(source))
            continue
        total += source.stat().st_size
    if missing:
        raise FileNotFoundError("Missing {} indexed frames, first: {}".format(len(missing), missing[0]))
    return total


def copy_subset(root, subset_root, split_result, min_free_gb, delete_source):
    frames = []
    for tracklet in split_result["tracklets"]:
        frames.extend(tracklet["frames"])
    unique_frames = sorted(set(frames))
    required = estimate_copy_bytes(root, unique_frames)
    available_after_copy = free_bytes(subset_root.parent) - required
    if available_after_copy < int(float(min_free_gb) * 1024 ** 3):
        raise RuntimeError(
            "Copying subset would leave {:.1f}GB free, below min-free-gb={:.1f}".format(
                available_after_copy / 1024 ** 3,
                float(min_free_gb),
            )
        )
    subset_root.mkdir(parents=True, exist_ok=True)
    for frame_rel in unique_frames:
        source = resolve_existing_frame(root, frame_rel)
        target = subset_root / frame_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        if delete_source:
            source.unlink()
    return {
        "subset_root": str(subset_root),
        "copied_frames": len(unique_frames),
        "copied_bytes": required,
        "delete_source": bool(delete_source),
    }


def check_required_files(root, split):
    required = [root / "data", root / "train_name.txt", root / "track_train_info.txt"]
    if split in ("test", "all"):
        required.extend([root / "test_name.txt", root / "track_test_info.txt", root / "query_IDX.txt"])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required HITSZ-VCM files/directories: {}".format(", ".join(missing)))


def build_meta(args, train_result, test_result=None, copy_info=None):
    tracklets = train_result["tracklets"]
    num_rgb = sum(1 for item in tracklets if item["modality"] == "rgb")
    num_ir = sum(1 for item in tracklets if item["modality"] == "ir")
    limited_frames = train_result["num_frames_limited"]
    meta = {
        "dataset": "HITSZ-VCM",
        "split": args.split,
        "root": str(Path(args.root).resolve()),
        "num_pids": train_result["num_pids"],
        "num_tracklets": train_result["num_tracklets"],
        "num_rgb_tracklets": num_rgb,
        "num_ir_tracklets": num_ir,
        "num_frames_original": train_result["num_frames_original"],
        "num_frames_limited": limited_frames,
        "num_frames_in_original_index": train_result["num_frames_original"],
        "num_frames_in_limited_index": limited_frames,
        "max_frames_per_tracklet": args.max_frames_per_tracklet,
        "sampling": args.sampling,
        "estimated_effective_images_per_epoch_for_K1": min(1, 4) * train_result["num_tracklets"],
        "estimated_effective_images_per_epoch_for_K2": min(2, 4) * train_result["num_tracklets"],
        "estimated_effective_images_per_epoch_for_K3": min(3, 4) * train_result["num_tracklets"],
        "estimated_effective_images_per_epoch_for_K4": min(4, 4) * train_result["num_tracklets"],
        "copy_subset": bool(args.copy_subset),
        "copy_info": copy_info,
        "academic_notice": ACADEMIC_NOTICE,
    }
    if test_result is not None:
        meta["test_num_tracklets"] = test_result["num_tracklets"]
        meta["test_num_frames_in_limited_index"] = test_result["num_frames_limited"]
    return meta


def main():
    parser = argparse.ArgumentParser(description="Safely prepare local HITSZ-VCM indexes")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "test", "all"), default="train")
    parser.add_argument("--max-frames-per-tracklet", type=int, default=16)
    parser.add_argument("--sampling", choices=("uniform", "random", "all"), default="uniform")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-subset", type=str_to_bool, default=False)
    parser.add_argument("--subset-root", default=None)
    parser.add_argument("--min-free-gb", type=float, default=30.0)
    parser.add_argument("--delete-source", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    check_required_files(root, args.split)
    require_min_free(output.parent, args.min_free_gb, "Index creation")
    if args.delete_source and not args.copy_subset:
        raise RuntimeError("--delete-source is only valid together with --copy-subset true")

    train_result = None
    test_result = None
    if args.split in ("train", "all"):
        train_result = build_split(
            root=root,
            split="train",
            name_file=root / "train_name.txt",
            info_file=root / "track_train_info.txt",
            max_frames=args.max_frames_per_tracklet,
            sampling=args.sampling,
            seed=args.seed,
        )
        write_json(output / "vcm_train_tracklets.json", {"metadata": train_result["metadata"], "tracklets": train_result["tracklets"]})
        write_jsonl(output / "vcm_train_frames.jsonl", train_result["frames"])
    if args.split in ("test", "all"):
        test_result = build_split(
            root=root,
            split="test",
            name_file=root / "test_name.txt",
            info_file=root / "track_test_info.txt",
            max_frames=args.max_frames_per_tracklet,
            sampling=args.sampling,
            seed=args.seed,
        )
        write_json(output / "vcm_test_tracklets.json", {"metadata": test_result["metadata"], "tracklets": test_result["tracklets"]})
        write_jsonl(output / "vcm_test_frames.jsonl", test_result["frames"])

    if train_result is None:
        train_result = test_result
    copy_info = None
    if args.copy_subset:
        subset_root = Path(args.subset_root).resolve() if args.subset_root else root.parent / "{}-subset".format(root.name)
        copy_info = copy_subset(
            root=root,
            subset_root=subset_root,
            split_result=train_result,
            min_free_gb=args.min_free_gb,
            delete_source=args.delete_source,
        )

    meta = build_meta(args, train_result=train_result, test_result=test_result, copy_info=copy_info)
    write_json(output / "vcm_meta.json", meta)

    print(ACADEMIC_NOTICE)
    print("Wrote indexes to {}".format(output))
    print("Tracklets: {} (rgb={}, ir={})".format(meta["num_tracklets"], meta["num_rgb_tracklets"], meta["num_ir_tracklets"]))
    print("Frames: original_index={} limited_index={}".format(meta["num_frames_in_original_index"], meta["num_frames_in_limited_index"]))
    print("Meta: {}".format(output / "vcm_meta.json"))


if __name__ == "__main__":
    main()
