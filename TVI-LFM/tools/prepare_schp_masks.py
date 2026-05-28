import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.sysumm01.datasets.schp_parts import resolve_schp_mask_path
from project.sysumm01.datasets.sysumm01 import build_test_records, build_train_records
from project.sysumm01.datasets.vcm import _load_tracklet_json, _resolve_frame_path


def parse_args():
    parser = argparse.ArgumentParser("Prepare offline SCHP masks in a source-mirrored directory.")
    parser.add_argument("--output-root", default="/home/cgv841/ybj/TVI-LFM/data/schp_masks/lip")
    parser.add_argument("--sysu-root", default="/home/cgv841/datasets/SYSU-MM01")
    parser.add_argument("--vcm-root", default="/home/cgv841/datasets/HITSZ-VCM")
    parser.add_argument(
        "--vcm-index",
        default="/home/cgv841/ybj/TVI-LFM/project/sysumm01/datasets/vcm_index/vcm_train_tracklets.json",
    )
    parser.add_argument("--schp-root", default="/home/cgv841/ybj/TVI-LFM/external/Self-Correction-Human-Parsing")
    parser.add_argument("--checkpoint", default="/home/cgv841/ybj/TVI-LFM/pretrained/schp/exp-schp-201908261155-lip.pth")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--include-sysu-train-ir", action="store_true")
    parser.add_argument("--include-sysu-test-ir", action="store_true")
    parser.add_argument("--include-vcm-ir", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def collect_sysu_train_ir(sysu_root):
    _, ir_records, _ = build_train_records(sysu_root, use_val=True, train_modality="ir")
    return [
        {
            "image_path": item["path"],
            "source_root": sysu_root,
            "source_name": "sysumm01",
        }
        for item in ir_records
    ]


def collect_sysu_test_ir(sysu_root):
    records = []
    for mode in ("all", "indoor"):
        query, gallery = build_test_records(sysu_root, mode=mode, protocol="same_modality", modality="ir")
        records.extend(query)
        records.extend(gallery)
    seen = set()
    items = []
    for item in records:
        path = item["path"]
        if path in seen:
            continue
        seen.add(path)
        items.append({"image_path": path, "source_root": sysu_root, "source_name": "sysumm01"})
    return items


def collect_vcm_ir(vcm_root, vcm_index):
    payload = _load_tracklet_json(vcm_index)
    seen = set()
    items = []
    for tracklet in payload["tracklets"]:
        if str(tracklet.get("modality", "")).lower() != "ir":
            continue
        for frame_path in tracklet.get("frames") or tracklet.get("frame_paths") or []:
            path = _resolve_frame_path(vcm_root, frame_path)
            if path in seen:
                continue
            seen.add(path)
            items.append({"image_path": path, "source_root": vcm_root, "source_name": "vcm"})
    return items


def stage_chunk(items, stage_dir):
    manifest = []
    input_dir = stage_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(items):
        src = Path(item["image_path"])
        suffix = src.suffix.lower() if src.suffix else ".jpg"
        staged_name = "{:07d}{}".format(index, suffix)
        staged_path = input_dir / staged_name
        os.symlink(src, staged_path)
        item = dict(item)
        item["staged_name"] = staged_name
        item["mask_name"] = Path(staged_name).with_suffix(".png").name
        manifest.append(item)
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


def copy_masks(manifest, schp_output_dir, output_root, overwrite=False):
    copied = 0
    for item in manifest:
        src = schp_output_dir / item["mask_name"]
        dst = Path(
            resolve_schp_mask_path(
                item["image_path"],
                output_root,
                item["source_root"],
                item["source_name"],
            )
        )
        if dst.exists() and not overwrite:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied


def main():
    args = parse_args()
    items = []
    if args.include_sysu_train_ir:
        items.extend(collect_sysu_train_ir(args.sysu_root))
    if args.include_sysu_test_ir:
        items.extend(collect_sysu_test_ir(args.sysu_root))
    if args.include_vcm_ir:
        items.extend(collect_vcm_ir(args.vcm_root, args.vcm_index))
    if not items:
        raise ValueError("No inputs selected. Enable at least one --include-* option.")

    filtered = []
    for item in items:
        dst = resolve_schp_mask_path(item["image_path"], args.output_root, item["source_root"], item["source_name"])
        if args.overwrite or not os.path.isfile(dst):
            filtered.append(item)
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.num_shards > 1:
        filtered = filtered[args.shard_index :: args.num_shards]

    print("Total source images: {}".format(len(items)), flush=True)
    print("Masks to generate: {}".format(len(filtered)), flush=True)
    print("Output root: {}".format(args.output_root), flush=True)
    print("Shard: {}/{}".format(args.shard_index, args.num_shards), flush=True)
    if args.dry_run or not filtered:
        return

    copied_total = 0
    for start in range(0, len(filtered), args.chunk_size):
        chunk = filtered[start : start + args.chunk_size]
        with tempfile.TemporaryDirectory(prefix="schp_masks_") as tmp:
            stage_dir = Path(tmp)
            input_dir, manifest = stage_chunk(chunk, stage_dir)
            mask_dir = stage_dir / "masks"
            run_schp(args, input_dir, mask_dir)
            copied_total += copy_masks(manifest, mask_dir, args.output_root, overwrite=args.overwrite)
        print("Processed {}/{} masks".format(min(start + args.chunk_size, len(filtered)), len(filtered)), flush=True)
    print("Copied masks: {}".format(copied_total), flush=True)


if __name__ == "__main__":
    main()
