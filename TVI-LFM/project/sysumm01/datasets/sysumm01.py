import os
import random
from collections import defaultdict
import glob
import re

import numpy as np
from PIL import Image
from torch.utils.data import BatchSampler, Dataset
from torchvision import transforms as T

from project.sysumm01.datasets.schp_parts import PairedImagePartTransform, load_part_mask, load_quality_index


RGB_CAMERAS = (1, 2, 4, 5)
IR_CAMERAS = (3, 6)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MSMT_FILENAME_RE = re.compile(r"([-\d]+)_c(\d+)")


def normalize_train_modality(train_modality):
    if train_modality in (None, "both"):
        return "both"
    if train_modality not in ("rgb", "ir"):
        raise ValueError("train_modality must be one of 'both', 'rgb', 'ir', got {}".format(train_modality))
    return train_modality


def get_eval_camera_ids(protocol, modality=None, mode="all"):
    if protocol == "cross_modality":
        if mode not in ("all", "indoor"):
            raise ValueError("mode must be 'all' or 'indoor', got {}".format(mode))
        query_cameras = IR_CAMERAS
        gallery_cameras = RGB_CAMERAS if mode == "all" else (1, 2)
        return query_cameras, gallery_cameras

    if protocol != "same_modality":
        raise ValueError("Unsupported protocol: {}".format(protocol))
    if modality not in ("rgb", "ir"):
        raise ValueError("same_modality evaluation requires modality='rgb' or 'ir', got {}".format(modality))
    if modality == "rgb":
        cameras = RGB_CAMERAS if mode == "all" else (1, 2)
    else:
        cameras = IR_CAMERAS
    return cameras, cameras


def read_id_file(path):
    with open(path, "r", encoding="utf-8") as handle:
        line = handle.read().strip().splitlines()[0]
    return [int(item) for item in line.split(",") if item.strip()]


def parse_record(path):
    path = path.replace("\\", "/")
    parts = path.split("/")
    cam_name = parts[-3]
    pid = int(parts[-2])
    camid = int(cam_name.replace("cam", ""))
    return pid, camid


def list_images_for_ids(root, camera_ids, person_ids):
    records = []
    for person_id in sorted(person_ids):
        identity = "{:04d}".format(person_id)
        for camid in camera_ids:
            folder = os.path.join(root, "cam{}".format(camid), identity)
            if not os.path.isdir(folder):
                continue
            for name in sorted(os.listdir(folder)):
                path = os.path.join(folder, name)
                if os.path.isfile(path):
                    pid, parsed_camid = parse_record(path)
                    records.append(
                        {
                            "path": path,
                            "pid": pid,
                            "camid": parsed_camid,
                            "modality": "rgb" if parsed_camid in RGB_CAMERAS else "ir",
                        }
                    )
    return records


def build_train_records(root, use_val=True, train_modality="both"):
    train_modality = normalize_train_modality(train_modality)
    train_ids = read_id_file(os.path.join(root, "exp", "train_id.txt"))
    if use_val:
        train_ids.extend(read_id_file(os.path.join(root, "exp", "val_id.txt")))

    rgb_records = list_images_for_ids(root, RGB_CAMERAS, train_ids)
    ir_records = list_images_for_ids(root, IR_CAMERAS, train_ids)

    if train_modality == "rgb":
        ir_records = []
    elif train_modality == "ir":
        rgb_records = []

    active_pids = sorted({record["pid"] for record in rgb_records + ir_records})
    pid_to_label = {pid: idx for idx, pid in enumerate(active_pids)}

    for record in rgb_records + ir_records:
        record["label"] = pid_to_label[record["pid"]]

    return rgb_records, ir_records, pid_to_label


def _resolve_msmt_root(root):
    candidates = [
        root,
        os.path.join(root, "MSMT17"),
        os.path.join(root, "MSMT17_V1"),
    ]
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "list_train.txt")) and os.path.isdir(os.path.join(candidate, "train")):
            return candidate, "list"
        if os.path.isdir(os.path.join(candidate, "bounding_box_train")):
            return candidate, "folders"
    raise RuntimeError(
        "MSMT17 root not found under {}. Expected either MSMT17/list_train.txt + train/ "
        "or MSMT17_V1/bounding_box_train/.".format(root)
    )


def _read_msmt_list_records(msmt_root, list_name):
    list_path = os.path.join(msmt_root, list_name)
    if not os.path.isfile(list_path):
        return []
    records = []
    with open(list_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            rel_path, pid_text = parts[:2]
            pid = int(pid_text)
            camid = int(rel_path.split("_")[2]) - 1
            records.append(
                {
                    "path": os.path.join(msmt_root, "train", rel_path),
                    "pid": pid,
                    "camid": camid,
                    "modality": "rgb",
                    "source": "msmt17",
                }
            )
    return records


def _read_msmt_folder_records(msmt_root):
    records = []
    train_dir = os.path.join(msmt_root, "bounding_box_train")
    for path in sorted(glob.glob(os.path.join(train_dir, "*.jpg"))):
        match = MSMT_FILENAME_RE.search(os.path.basename(path))
        if not match:
            continue
        pid, camid = map(int, match.groups())
        if pid == -1:
            continue
        records.append(
            {
                "path": path,
                "pid": pid,
                "camid": camid - 1,
                "modality": "rgb",
                "source": "msmt17",
            }
        )
    return records


def build_msmt_train_records(root, use_val=True):
    msmt_root, layout = _resolve_msmt_root(root)
    if layout == "list":
        records = _read_msmt_list_records(msmt_root, "list_train.txt")
        if use_val:
            records.extend(_read_msmt_list_records(msmt_root, "list_val.txt"))
    else:
        records = _read_msmt_folder_records(msmt_root)

    records = [record for record in records if os.path.isfile(record["path"])]
    if not records:
        raise RuntimeError("No MSMT17 training images found under {}".format(msmt_root))
    return records, msmt_root, layout


class MixedRGBTrainDataset(Dataset):
    """RGB-only training set mixed from MSMT17 train split and SYSU-MM01 train IDs.

    The dataset deliberately never reads MSMT17 query/gallery/test folders or
    SYSU-MM01 exp/test_id.txt, so evaluation identities cannot leak into
    pretraining.
    """

    def __init__(
        self,
        sysu_root,
        msmt_root,
        image_size,
        sysu_use_val=True,
        msmt_use_val=True,
        train_augment="basic",
    ):
        sysu_rgb_records, _, _ = build_train_records(
            sysu_root,
            use_val=sysu_use_val,
            train_modality="rgb",
        )
        for record in sysu_rgb_records:
            record["source"] = "sysumm01"

        msmt_records, resolved_msmt_root, msmt_layout = build_msmt_train_records(
            msmt_root,
            use_val=msmt_use_val,
        )

        source_records = [
            ("sysumm01", sysu_rgb_records),
            ("msmt17", msmt_records),
        ]
        self.records = []
        self.source_counts = {}
        label_offset = 0
        for source, records in source_records:
            active_pids = sorted({record["pid"] for record in records})
            pid_to_label = {pid: label_offset + idx for idx, pid in enumerate(active_pids)}
            self.source_counts[source] = {
                "images": len(records),
                "pids": len(active_pids),
            }
            for record in records:
                item = dict(record)
                item["label"] = pid_to_label[item["pid"]]
                self.records.append(item)
            label_offset += len(active_pids)

        self.num_classes = label_offset
        self.transform = build_transforms(image_size=image_size, training=True, augment=train_augment)
        self.indices_by_label = defaultdict(list)
        for index, record in enumerate(self.records):
            self.indices_by_label[record["label"]].append(index)
        self.valid_labels = sorted(self.indices_by_label.keys())
        self.resolved_roots = {
            "sysumm01": sysu_root,
            "msmt17": resolved_msmt_root,
            "msmt17_layout": msmt_layout,
        }

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(record["path"]).convert("RGB")
        tensor = self.transform(image)
        return {
            "image": tensor,
            "label": record["label"],
            "pid": record["pid"],
            "camid": record["camid"],
            "modality": 0,
            "source": record["source"],
            "path": record["path"],
        }


def build_test_records(root, mode="all", protocol="cross_modality", modality=None):
    test_ids = read_id_file(os.path.join(root, "exp", "test_id.txt"))
    query_cameras, gallery_cameras = get_eval_camera_ids(protocol=protocol, modality=modality, mode=mode)
    query_records = list_images_for_ids(root, query_cameras, test_ids)
    gallery_records = list_images_for_ids(root, gallery_cameras, test_ids)
    return query_records, gallery_records


def build_transforms(image_size, training, augment="basic"):
    height, width = image_size
    ops = [T.Resize((height, width), interpolation=Image.BICUBIC)]
    if training:
        if augment == "basic":
            ops.append(T.RandomHorizontalFlip(p=0.5))
        elif augment == "strong_reid":
            ops.extend(
                [
                    T.Pad(10),
                    T.RandomCrop((height, width)),
                    T.RandomHorizontalFlip(p=0.5),
                ]
            )
        else:
            raise ValueError("Unsupported training augment: {}".format(augment))
    ops.extend([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    if training and augment == "strong_reid":
        ops.append(T.RandomErasing(p=0.5, value=IMAGENET_MEAN))
    return T.Compose(ops)


class SYSUTrainDataset(Dataset):
    def __init__(self, root, image_size, use_val=True, train_augment="basic", train_modality="both"):
        self.train_modality = normalize_train_modality(train_modality)
        rgb_records, ir_records, pid_to_label = build_train_records(
            root,
            use_val=use_val,
            train_modality=self.train_modality,
        )
        self.records = rgb_records + ir_records
        self.transform = build_transforms(image_size=image_size, training=True, augment=train_augment)
        self.num_classes = len(pid_to_label)
        self.indices_by_label = defaultdict(list)
        self.rgb_by_pid = defaultdict(list)
        self.ir_by_pid = defaultdict(list)
        for index, record in enumerate(self.records):
            self.indices_by_label[record["label"]].append(index)
            if record["modality"] == "rgb":
                self.rgb_by_pid[record["label"]].append(index)
            else:
                self.ir_by_pid[record["label"]].append(index)
        if self.train_modality == "both":
            self.valid_labels = sorted(set(self.rgb_by_pid.keys()) & set(self.ir_by_pid.keys()))
        else:
            self.valid_labels = sorted(self.indices_by_label.keys())

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(record["path"]).convert("RGB")
        tensor = self.transform(image)
        return {
            "image": tensor,
            "label": record["label"],
            "pid": record["pid"],
            "camid": record["camid"],
            "modality": 0 if record["modality"] == "rgb" else 1,
            "path": record["path"],
        }


class SYSUEvalDataset(Dataset):
    def __init__(
        self,
        records,
        image_size,
        schp_mask_root=None,
        schp_source_root=None,
        schp_min_part_pixels=4,
        schp_allow_fallback=True,
        schp_quality_index=None,
    ):
        self.records = records
        self.schp_mask_root = schp_mask_root
        self.schp_source_root = schp_source_root
        self.schp_min_part_pixels = int(schp_min_part_pixels)
        self.schp_allow_fallback = bool(schp_allow_fallback)
        self.schp_quality_index = load_quality_index(schp_quality_index)
        self.use_part_masks = schp_mask_root is not None
        if self.use_part_masks:
            self.transform = PairedImagePartTransform(image_size=image_size, training=False, augment="basic")
        else:
            self.transform = build_transforms(image_size=image_size, training=False)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(record["path"]).convert("RGB")
        if self.use_part_masks:
            part_mask = load_part_mask(
                record["path"],
                image_size=image.size,
                mask_root=self.schp_mask_root,
                source_root=self.schp_source_root,
                source_name="sysumm01",
                min_part_pixels=self.schp_min_part_pixels,
                allow_fallback=self.schp_allow_fallback,
                quality_index=self.schp_quality_index,
            )
            tensor, part_mask_tensor = self.transform(image, part_mask)
        else:
            tensor = self.transform(image)
            part_mask_tensor = None
        result = {
            "image": tensor,
            "pid": record["pid"],
            "camid": record["camid"],
            "modality": 0 if record["modality"] == "rgb" else 1,
            "path": record["path"],
        }
        if part_mask_tensor is not None:
            result["part_masks"] = part_mask_tensor
        return result


class CrossModalBatchSampler(BatchSampler):
    def __init__(self, dataset, num_ids, num_instances, num_batches, seed=42, rgb_instances=None, ir_instances=None):
        self.dataset = dataset
        self.num_ids = num_ids
        self.num_instances = num_instances
        self.rgb_instances = int(rgb_instances if rgb_instances is not None else num_instances)
        self.ir_instances = int(ir_instances if ir_instances is not None else num_instances)
        self.num_batches = num_batches
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return self.num_batches

    @staticmethod
    def _sample(indices, count, rng):
        if len(indices) >= count:
            return rng.sample(indices, count)
        return [rng.choice(indices) for _ in range(count)]

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.num_batches):
            chosen_ids = rng.sample(self.dataset.valid_labels, self.num_ids)
            batch = []
            for label in chosen_ids:
                batch.extend(self._sample(self.dataset.rgb_by_pid[label], self.rgb_instances, rng))
                batch.extend(self._sample(self.dataset.ir_by_pid[label], self.ir_instances, rng))
            rng.shuffle(batch)
            yield batch


class IdentityBatchSampler(BatchSampler):
    def __init__(self, dataset, num_ids, num_instances, num_batches, seed=42):
        self.dataset = dataset
        self.num_ids = num_ids
        self.num_instances = num_instances
        self.num_batches = num_batches
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return self.num_batches

    @staticmethod
    def _sample(indices, count, rng):
        if len(indices) >= count:
            return rng.sample(indices, count)
        return [rng.choice(indices) for _ in range(count)]

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.num_batches):
            chosen_ids = rng.sample(self.dataset.valid_labels, self.num_ids)
            batch = []
            for label in chosen_ids:
                batch.extend(self._sample(self.dataset.indices_by_label[label], self.num_instances, rng))
            rng.shuffle(batch)
            yield batch



class FullCoverageIdentityBatchSampler(BatchSampler):
    """Identity-balanced sampler with full-coverage priority."""

    def __init__(self, dataset, num_ids, num_instances, num_batches=None, seed=42, min_coverage=0.75):
        self.dataset = dataset
        self.num_ids = num_ids
        self.num_instances = num_instances
        self.seed = seed
        self.epoch = 0
        self.min_coverage = float(min_coverage)
        self.batch_size = self.num_ids * self.num_instances
        target_batches = int(np.ceil(len(self.dataset) * self.min_coverage / float(self.batch_size)))
        self.num_batches = max(int(num_batches or 0), max(1, target_batches))

    def __len__(self):
        return self.num_batches

    def _draw_instances(self, label, remaining_by_label, full_by_label, rng):
        remaining = remaining_by_label[label]
        if len(remaining) >= self.num_instances:
            picked = remaining[: self.num_instances]
            del remaining[: self.num_instances]
            return picked

        picked = list(remaining)
        remaining.clear()
        pool = list(full_by_label[label])
        rng.shuffle(pool)
        needed = self.num_instances - len(picked)
        if len(pool) >= needed:
            picked.extend(pool[:needed])
            remaining.extend(pool[needed:])
        else:
            picked.extend(pool)
            while len(picked) < self.num_instances:
                picked.append(rng.choice(full_by_label[label]))
        return picked

    def _choose_ids(self, labels, remaining_by_label, rng):
        # Prefer identities that still have enough unseen samples for PK draw.
        rich = [label for label in labels if len(remaining_by_label[label]) >= self.num_instances]
        poor = [label for label in labels if 0 < len(remaining_by_label[label]) < self.num_instances]

        rich.sort(key=lambda label: len(remaining_by_label[label]), reverse=True)
        poor.sort(key=lambda label: len(remaining_by_label[label]), reverse=True)

        chosen = []
        if rich:
            chosen.extend(rich[: self.num_ids])

        if len(chosen) < self.num_ids and poor:
            needed = self.num_ids - len(chosen)
            chosen.extend(poor[:needed])

        if len(chosen) < self.num_ids:
            all_labels = list(labels)
            rng.shuffle(all_labels)
            for label in all_labels:
                if len(chosen) >= self.num_ids:
                    break
                if label not in chosen:
                    chosen.append(label)

        return chosen[: self.num_ids]

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        labels = list(self.dataset.valid_labels)
        rng.shuffle(labels)

        full_by_label = {}
        remaining_by_label = {}
        for label in labels:
            indices = list(self.dataset.indices_by_label[label])
            rng.shuffle(indices)
            full_by_label[label] = indices
            remaining_by_label[label] = list(indices)

        for _ in range(self.num_batches):
            chosen_ids = self._choose_ids(labels, remaining_by_label, rng)

            batch = []
            for label in chosen_ids:
                batch.extend(self._draw_instances(label, remaining_by_label, full_by_label, rng))
            rng.shuffle(batch)
            yield batch


def l2_normalize(array):
    denom = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(denom, a_min=1e-12, a_max=None)
