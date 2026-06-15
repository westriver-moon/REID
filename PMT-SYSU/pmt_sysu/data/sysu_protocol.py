from __future__ import annotations

import random
from pathlib import Path

import numpy as np


def _read_test_ids(root: Path):
    with (root / "exp" / "test_id.txt").open("r", encoding="utf-8") as handle:
        line = handle.read().splitlines()[0]
    return [f"{int(value):04d}" for value in line.split(",")]


def _parse_cam_pid(path: Path) -> tuple[int, int]:
    return int(path.parent.parent.name.replace("cam", "")), int(path.parent.name)


def process_query_sysu(data_path: str | Path, mode: str = "all"):
    root = Path(data_path)
    ir_cameras = ["cam3", "cam6"] if mode in {"all", "indoor"} else None
    if ir_cameras is None:
        raise ValueError(f"Unsupported SYSU query mode: {mode}")
    files_ir = []
    for pid in _read_test_ids(root):
        for cam in ir_cameras:
            img_dir = root / cam / pid
            if img_dir.is_dir():
                files_ir.extend(sorted(img_dir.iterdir()))

    query_img, query_id, query_cam = [], [], []
    for img_path in files_ir:
        camid, pid = _parse_cam_pid(img_path)
        query_img.append(str(img_path))
        query_id.append(pid)
        query_cam.append(camid)
    return query_img, np.asarray(query_id), np.asarray(query_cam)


def process_gallery_sysu(data_path: str | Path, mode: str = "all", trial: int = 0, gall_mode: str = "single"):
    root = Path(data_path)
    random.seed(trial)
    if mode == "all":
        rgb_cameras = ["cam1", "cam2", "cam4", "cam5"]
    elif mode == "indoor":
        rgb_cameras = ["cam1", "cam2"]
    else:
        raise ValueError(f"Unsupported SYSU gallery mode: {mode}")

    files_rgb = []
    for pid in _read_test_ids(root):
        for cam in rgb_cameras:
            img_dir = root / cam / pid
            if img_dir.is_dir():
                new_files = sorted(img_dir.iterdir())
                if not new_files:
                    continue
                if gall_mode == "single":
                    files_rgb.append(random.choice(new_files))
                elif gall_mode == "multi":
                    replace = len(new_files) < 10
                    files_rgb.extend(np.random.choice(new_files, 10, replace=replace).tolist())
                else:
                    raise ValueError(f"Unsupported gallery mode: {gall_mode}")

    gall_img, gall_id, gall_cam = [], [], []
    for img_path in files_rgb:
        camid, pid = _parse_cam_pid(Path(img_path))
        gall_img.append(str(img_path))
        gall_id.append(pid)
        gall_cam.append(camid)
    return gall_img, np.asarray(gall_id), np.asarray(gall_cam)

