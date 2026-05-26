#!/usr/bin/env python
import argparse
import json
import os
import re
from pathlib import Path


CAMERA_RE = re.compile(r"(?:^|[_/\-])c(?:am)?(\d+)(?:[_/\-]|$)", re.IGNORECASE)


def read_lines(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def parse_int_list(text):
    if text is None or text == "":
        return set()
    return {int(item) for item in text.replace(",", " ").split()}


def parse_camera_from_path(path):
    match = CAMERA_RE.search(path.replace("\\", "/"))
    if match:
        return int(match.group(1))
    return None


def infer_modality(path, camid, rgb_camera_ids, ir_camera_ids):
    lowered = path.lower().replace("\\", "/")
    parts = re.split(r"[/_\-.]+", lowered)
    if any(part in ("rgb", "visible", "vis", "color") for part in parts):
        return "rgb"
    if any(part in ("ir", "infrared", "thermal") for part in parts):
        return "ir"
    if camid in rgb_camera_ids:
        return "rgb"
    if camid in ir_camera_ids:
        return "ir"
    raise ValueError(
        "Unable to infer modality for path={} camid={}. Pass --rgb-camera-ids and --ir-camera-ids.".format(
            path,
            camid,
        )
    )


def normalize_frame_path(root, frame_path):
    frame_path = frame_path.replace("\\", "/")
    if os.path.isabs(frame_path):
        try:
            return os.path.relpath(frame_path, root).replace("\\", "/")
        except ValueError:
            return frame_path
    return frame_path


def parse_track_info_line(line, line_no):
    tokens = line.replace(",", " ").split()
    numbers = []
    for token in tokens:
        try:
            numbers.append(int(token))
        except ValueError:
            continue
    if len(numbers) < 4:
        raise ValueError("track_train_info.txt line {} must contain at least 4 integers: {}".format(line_no, line))
    return numbers[0], numbers[1], numbers[2], numbers[3], numbers[4:]


def build_tracklets(args):
    root = os.path.abspath(args.root)
    train_name_path = args.train_name or find_default_file(root, "train_name.txt")
    track_info_path = args.track_info or find_default_file(root, "track_train_info.txt")

    frame_names = read_lines(train_name_path)
    info_lines = read_lines(track_info_path)
    rgb_camera_ids = parse_int_list(args.rgb_camera_ids)
    ir_camera_ids = parse_int_list(args.ir_camera_ids)

    parsed = []
    starts = []
    ends = []
    for line_no, line in enumerate(info_lines, start=1):
        start, end, pid, camid, extra = parse_track_info_line(line, line_no)
        parsed.append((start, end, pid, camid, extra))
        starts.append(start)
        ends.append(end)

    if not parsed:
        raise RuntimeError("No tracklets found in {}".format(track_info_path))

    one_based = min(starts) >= 1 and max(ends) <= len(frame_names)
    tracklets = []
    label_pids = sorted({pid for _, _, pid, _, _ in parsed})
    pid_to_label = {pid: idx for idx, pid in enumerate(label_pids)}

    for tracklet_id, (start, end, pid, camid, extra) in enumerate(parsed):
        begin = start - 1 if one_based else start
        finish = end if one_based else end + 1
        if begin < 0 or finish > len(frame_names) or begin >= finish:
            raise ValueError(
                "Invalid frame range at tracklet {}: start={} end={} with {} frames".format(
                    tracklet_id,
                    start,
                    end,
                    len(frame_names),
                )
            )
        frames = [normalize_frame_path(root, item) for item in frame_names[begin:finish]]
        inferred_camid = camid
        if inferred_camid < 0:
            parsed_camid = parse_camera_from_path(frames[0])
            if parsed_camid is None:
                raise ValueError("Unable to parse camid for tracklet {}".format(tracklet_id))
            inferred_camid = parsed_camid
        modality = infer_modality(frames[0], inferred_camid, rgb_camera_ids, ir_camera_ids)
        tracklets.append(
            {
                "tracklet_id": tracklet_id,
                "pid": pid,
                "label": pid_to_label[pid],
                "camid": inferred_camid,
                "modality": modality,
                "frames": frames,
                "num_frames": len(frames),
                "source_range": [start, end],
                "extra": extra,
            }
        )

    return {
        "metadata": {
            "dataset": "HITSZ-VCM",
            "split": args.split,
            "root": root,
            "train_name": os.path.abspath(train_name_path),
            "track_train_info": os.path.abspath(track_info_path),
            "index_base": 1 if one_based else 0,
            "num_frames": len(frame_names),
            "num_tracklets": len(tracklets),
            "num_pids": len(label_pids),
            "rgb_camera_ids": sorted(rgb_camera_ids),
            "ir_camera_ids": sorted(ir_camera_ids),
        },
        "tracklets": tracklets,
    }


def find_default_file(root, name):
    candidates = [
        os.path.join(root, name),
        os.path.join(root, "info", name),
        os.path.join(root, "Info", name),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError("Could not find {} under {} or {}/info".format(name, root, root))


def main():
    parser = argparse.ArgumentParser(description="Prepare HITSZ-VCM train tracklet metadata")
    parser.add_argument("--root", required=True, help="HITSZ-VCM root directory")
    parser.add_argument("--train-name", default=None, help="Path to train_name.txt; defaults to <root>/train_name.txt")
    parser.add_argument(
        "--track-info",
        default=None,
        help="Path to track_train_info.txt; defaults to <root>/track_train_info.txt",
    )
    parser.add_argument("--output", default=None, help="Output json path; defaults to <root>/vcm_train_tracklets.json")
    parser.add_argument("--split", default="train")
    parser.add_argument("--rgb-camera-ids", default="", help="Comma or space separated visible camera ids")
    parser.add_argument("--ir-camera-ids", default="", help="Comma or space separated infrared camera ids")
    args = parser.parse_args()

    payload = build_tracklets(args)
    output = args.output or os.path.join(os.path.abspath(args.root), "vcm_train_tracklets.json")
    Path(os.path.dirname(output)).mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    meta = payload["metadata"]
    rgb = sum(1 for item in payload["tracklets"] if item["modality"] == "rgb")
    ir = sum(1 for item in payload["tracklets"] if item["modality"] == "ir")
    print(
        "Wrote {} tracklets ({} rgb, {} ir, {} pids, {} frames) to {}".format(
            meta["num_tracklets"],
            rgb,
            ir,
            meta["num_pids"],
            meta["num_frames"],
            output,
        )
    )


if __name__ == "__main__":
    main()
