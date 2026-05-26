import json
import os
import random
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import BatchSampler, Dataset

from project.sysumm01.datasets.sysumm01 import build_train_records, build_transforms


MODALITY_TO_ID = {"rgb": 0, "ir": 1}


def normalize_vcm_mode(mode):
    if mode not in ("rgb_ir", "rgb_only", "ir_only"):
        raise ValueError("mode must be one of rgb_ir, rgb_only, ir_only, got {}".format(mode))
    return mode


def _load_tracklet_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "tracklets" in payload:
        return payload
    if isinstance(payload, list):
        return {"tracklets": payload, "metadata": {}}
    raise ValueError("Invalid VCM tracklet json: {}".format(path))


def _resolve_frame_path(root, frame_path):
    if os.path.isabs(frame_path):
        return frame_path
    return os.path.join(root, frame_path)


def _sample(indices, count, rng):
    if len(indices) >= count:
        return rng.sample(indices, count)
    return [rng.choice(indices) for _ in range(count)]


def _uniform_sample(frames, count):
    if len(frames) >= count:
        if count == 1:
            return [frames[len(frames) // 2]]
        positions = [
            round(index * (len(frames) - 1) / float(count - 1))
            for index in range(count)
        ]
        return [frames[int(pos)] for pos in positions]
    return [frames[index % len(frames)] for index in range(count)]


class VCMTrackletDataset(Dataset):
    """HITSZ-VCM tracklet-level dataset.

    Each sample is one tracklet. Frames are sampled inside __getitem__, so the
    training sampler always works on tracklet indices rather than flattened
    image indices.
    """

    def __init__(
        self,
        root=None,
        tracklet_json=None,
        image_size=(288, 144),
        frames_per_tracklet=2,
        mode="rgb_ir",
        frame_sampling="random",
        train_augment="strong_reid",
        index_path=None,
    ):
        tracklet_json = tracklet_json or index_path
        if tracklet_json is None:
            raise ValueError("VCMTrackletDataset requires tracklet_json or index_path")
        payload = _load_tracklet_json(tracklet_json)
        root = root or payload.get("metadata", {}).get("root")
        if root is None:
            raise ValueError("VCMTrackletDataset requires root when index metadata has no root")
        self.root = root
        self.tracklet_json = tracklet_json
        self.mode = normalize_vcm_mode(mode)
        self.frame_sampling = frame_sampling
        if self.frame_sampling not in ("random", "uniform"):
            raise ValueError("frame_sampling must be random or uniform, got {}".format(frame_sampling))
        self.frames_per_tracklet = int(frames_per_tracklet)
        if self.frames_per_tracklet < 1 or self.frames_per_tracklet > 4:
            raise ValueError("frames_per_tracklet must be in [1, 4], got {}".format(self.frames_per_tracklet))

        raw_tracklets = payload["tracklets"]
        allowed_modalities = {
            "rgb_ir": {"rgb", "ir"},
            "rgb_only": {"rgb"},
            "ir_only": {"ir"},
        }[self.mode]

        filtered = []
        for item in raw_tracklets:
            modality = str(item["modality"]).lower()
            if modality not in allowed_modalities:
                continue
            frames = item.get("frames") or item.get("frame_paths")
            if not frames:
                continue
            filtered.append(
                {
                    "tracklet_id": item.get("tracklet_id", len(filtered)),
                    "pid": int(item["pid"]),
                    "camid": int(item["camid"]),
                    "modality": modality,
                    "frames": list(frames),
                }
            )

        if not filtered:
            raise RuntimeError("No VCM tracklets found for mode={} in {}".format(self.mode, tracklet_json))

        active_pids = sorted({item["pid"] for item in filtered})
        pid_to_label = {pid: idx for idx, pid in enumerate(active_pids)}

        self.tracklets = []
        self.indices_by_label = defaultdict(list)
        self.rgb_by_label = defaultdict(list)
        self.ir_by_label = defaultdict(list)

        for index, item in enumerate(filtered):
            item = dict(item)
            item["label"] = pid_to_label[item["pid"]]
            self.tracklets.append(item)
            self.indices_by_label[item["label"]].append(index)
            if item["modality"] == "rgb":
                self.rgb_by_label[item["label"]].append(index)
            else:
                self.ir_by_label[item["label"]].append(index)

        if self.mode == "rgb_ir":
            self.valid_labels = sorted(set(self.rgb_by_label.keys()) & set(self.ir_by_label.keys()))
        else:
            self.valid_labels = sorted(self.indices_by_label.keys())
        if not self.valid_labels:
            raise RuntimeError("No valid identities available for VCM mode={}".format(self.mode))

        self.num_classes = len(active_pids)
        self.transform = build_transforms(image_size=image_size, training=True, augment=train_augment)
        self.metadata = payload.get("metadata", {})
        self.source_counts = self._count_sources()

    def _count_sources(self):
        counts = {
            "tracklets": len(self.tracklets),
            "pids": self.num_classes,
            "rgb_tracklets": 0,
            "ir_tracklets": 0,
            "frames": 0,
        }
        for item in self.tracklets:
            if item["modality"] == "rgb":
                counts["rgb_tracklets"] += 1
            else:
                counts["ir_tracklets"] += 1
            counts["frames"] += len(item["frames"])
        return counts

    def __len__(self):
        return len(self.tracklets)

    def _sample_frames(self, frames):
        if self.frame_sampling == "uniform":
            return _uniform_sample(frames, self.frames_per_tracklet)
        if len(frames) >= self.frames_per_tracklet:
            return random.sample(frames, self.frames_per_tracklet)
        return [random.choice(frames) for _ in range(self.frames_per_tracklet)]

    def __getitem__(self, index):
        item = self.tracklets[index]
        sampled_frames = self._sample_frames(item["frames"])
        images = []
        resolved_paths = []
        for frame_path in sampled_frames:
            full_path = _resolve_frame_path(self.root, frame_path)
            image = Image.open(full_path).convert("RGB")
            images.append(self.transform(image))
            resolved_paths.append(full_path)

        return {
            "images": torch.stack(images, dim=0),
            "label": item["label"],
            "pid": item["pid"],
            "camid": item["camid"],
            "modality": MODALITY_TO_ID[item["modality"]],
            "tracklet_id": item["tracklet_id"],
            "frame_paths": resolved_paths,
        }


class IdentityModalityBalancedTrackletSampler(BatchSampler):
    """PK sampler over tracklet indices with optional RGB/IR balance."""

    def __init__(
        self,
        dataset,
        num_ids,
        num_rgb_tracklets,
        num_ir_tracklets,
        num_batches,
        seed=42,
    ):
        self.dataset = dataset
        self.num_ids = int(num_ids)
        self.num_rgb_tracklets = int(num_rgb_tracklets)
        self.num_ir_tracklets = int(num_ir_tracklets)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.epoch = 0
        self.mode = dataset.mode

        if self.num_ids <= 0 or self.num_batches <= 0:
            raise ValueError("num_ids and num_batches must be positive")
        if self.mode == "rgb_ir" and (self.num_rgb_tracklets <= 0 or self.num_ir_tracklets <= 0):
            raise ValueError("rgb_ir mode requires positive RGB and IR tracklets per identity")
        if self.mode == "rgb_only" and self.num_rgb_tracklets <= 0:
            raise ValueError("rgb_only mode requires positive RGB tracklets per identity")
        if self.mode == "ir_only" and self.num_ir_tracklets <= 0:
            raise ValueError("ir_only mode requires positive IR tracklets per identity")

    def __len__(self):
        return self.num_batches

    def _sample_for_label(self, label, rng):
        if self.mode == "rgb_ir":
            indices = []
            indices.extend(_sample(self.dataset.rgb_by_label[label], self.num_rgb_tracklets, rng))
            indices.extend(_sample(self.dataset.ir_by_label[label], self.num_ir_tracklets, rng))
            return indices
        if self.mode == "rgb_only":
            return _sample(self.dataset.rgb_by_label[label], self.num_rgb_tracklets, rng)
        return _sample(self.dataset.ir_by_label[label], self.num_ir_tracklets, rng)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        labels = list(self.dataset.valid_labels)
        if len(labels) < self.num_ids:
            raise RuntimeError(
                "VCM sampler needs at least {} valid identities, got {}".format(self.num_ids, len(labels))
            )

        for _ in range(self.num_batches):
            chosen_labels = rng.sample(labels, self.num_ids)
            batch = []
            for label in chosen_labels:
                batch.extend(self._sample_for_label(label, rng))
            rng.shuffle(batch)
            yield batch


class SYSUIRVCMIRDataset(Dataset):
    """Image-level SYSU-MM01 IR target data plus HITSZ-VCM IR tracklet supplements.

    SYSU-MM01 IR samples stay as ordinary images. HITSZ-VCM IR samples stay as
    tracklets internally, and __getitem__ draws one frame from the selected
    tracklet so training does not flatten every VCM frame into a static image
    list.
    """

    def __init__(
        self,
        vcm_root,
        vcm_tracklet_json,
        sysu_root,
        image_size,
        sysu_use_val=True,
        vcm_frame_sampling="random",
        vcm_frames_per_tracklet=1,
        train_augment="strong_reid",
    ):
        self.vcm_root = vcm_root
        self.sysu_root = sysu_root
        self.tracklet_json = vcm_tracklet_json
        self.mode = "sysu_ir_vcm_ir"
        self.vcm_frame_sampling = vcm_frame_sampling
        if self.vcm_frame_sampling not in ("random", "uniform"):
            raise ValueError("vcm_frame_sampling must be random or uniform, got {}".format(vcm_frame_sampling))
        self.vcm_frames_per_tracklet = int(vcm_frames_per_tracklet)
        if self.vcm_frames_per_tracklet < 1 or self.vcm_frames_per_tracklet > 4:
            raise ValueError(
                "vcm_frames_per_tracklet must be in [1, 4], got {}".format(self.vcm_frames_per_tracklet)
            )

        self.transform = build_transforms(image_size=image_size, training=True, augment=train_augment)
        self.samples = []
        self.indices_by_label = defaultdict(list)
        self.ir_by_label = defaultdict(list)
        self.vcm_ir_by_label = defaultdict(list)
        self.sysu_ir_by_label = defaultdict(list)

        _, sysu_ir_records, sysu_pid_to_local_label = build_train_records(
            sysu_root,
            use_val=sysu_use_val,
            train_modality="ir",
        )
        if not sysu_ir_records:
            raise RuntimeError("No SYSU-MM01 IR records found under {}".format(sysu_root))
        for record in sysu_ir_records:
            self._append_sample(
                {
                    "source": "sysumm01_ir",
                    "sample_type": "image",
                    "pid": int(record["pid"]),
                    "label": int(record["label"]),
                    "camid": int(record["camid"]),
                    "modality": "ir",
                    "path": record["path"],
                    "tracklet_id": "sysu_ir:{}".format(len(self.samples)),
                }
            )

        sysu_class_count = len(sysu_pid_to_local_label)
        vcm_payload = _load_tracklet_json(vcm_tracklet_json)
        vcm_raw = []
        for item in vcm_payload["tracklets"]:
            modality = str(item["modality"]).lower()
            if modality != "ir":
                continue
            frames = item.get("frames") or item.get("frame_paths")
            if not frames:
                continue
            vcm_raw.append(
                {
                    "source": "vcm",
                    "sample_type": "tracklet",
                    "tracklet_id": "vcm:{}".format(item.get("tracklet_id", len(vcm_raw))),
                    "pid": int(item["pid"]),
                    "camid": int(item["camid"]),
                    "modality": "ir",
                    "frames": list(frames),
                }
            )
        if not vcm_raw:
            raise RuntimeError("No VCM IR tracklets found in {}".format(vcm_tracklet_json))

        vcm_pids = sorted({item["pid"] for item in vcm_raw})
        vcm_pid_to_label = {pid: sysu_class_count + idx for idx, pid in enumerate(vcm_pids)}
        for item in vcm_raw:
            item["label"] = vcm_pid_to_label[item["pid"]]
            self._append_sample(item)

        self.num_classes = sysu_class_count + len(vcm_pids)
        self.vcm_ir_valid_labels = sorted(self.vcm_ir_by_label.keys())
        self.sysu_ir_valid_labels = sorted(self.sysu_ir_by_label.keys())
        self.valid_labels = sorted(self.indices_by_label.keys())
        if not self.vcm_ir_valid_labels:
            raise RuntimeError("No VCM IR identities available")
        if not self.sysu_ir_valid_labels:
            raise RuntimeError("No SYSU-MM01 IR identities available")
        self.metadata = {
            "vcm": vcm_payload.get("metadata", {}),
            "sysu_use_val": sysu_use_val,
            "sysu_classes": sysu_class_count,
            "vcm_ir_classes": len(vcm_pids),
        }
        self.source_counts = self._count_sources()

    def _append_sample(self, item):
        index = len(self.samples)
        self.samples.append(item)
        label = item["label"]
        self.indices_by_label[label].append(index)
        self.ir_by_label[label].append(index)
        if item["source"] == "vcm":
            self.vcm_ir_by_label[label].append(index)
        elif item["source"] == "sysumm01_ir":
            self.sysu_ir_by_label[label].append(index)

    def _count_sources(self):
        counts = {
            "images_or_tracklets": len(self.samples),
            "pids": self.num_classes,
            "sysu_ir_images": 0,
            "vcm_ir_tracklets": 0,
            "vcm_ir_index_frames": 0,
            "vcm_ir_pids": len(self.vcm_ir_valid_labels),
            "sysu_ir_pids": len(self.sysu_ir_valid_labels),
        }
        for item in self.samples:
            if item["source"] == "vcm":
                counts["vcm_ir_tracklets"] += 1
                counts["vcm_ir_index_frames"] += len(item["frames"])
            elif item["source"] == "sysumm01_ir":
                counts["sysu_ir_images"] += 1
        return counts

    def __len__(self):
        return len(self.samples)

    def _sample_vcm_frames(self, frames):
        if self.vcm_frame_sampling == "uniform":
            return _uniform_sample(frames, self.vcm_frames_per_tracklet)
        if len(frames) >= self.vcm_frames_per_tracklet:
            return random.sample(frames, self.vcm_frames_per_tracklet)
        return [random.choice(frames) for _ in range(self.vcm_frames_per_tracklet)]

    def __getitem__(self, index):
        item = self.samples[index]
        if item["sample_type"] == "tracklet":
            frame_paths = self._sample_vcm_frames(item["frames"])
            full_paths = [_resolve_frame_path(self.vcm_root, frame_path) for frame_path in frame_paths]
        else:
            full_paths = [item["path"]]
        images = []
        for full_path in full_paths:
            image = Image.open(full_path).convert("RGB")
            images.append(self.transform(image))
        return {
            "images": torch.stack(images, dim=0),
            "label": item["label"],
            "pid": item["pid"],
            "camid": item["camid"],
            "modality": MODALITY_TO_ID[item["modality"]],
            "tracklet_id": item["tracklet_id"],
            "source": item["source"],
            "frame_paths": full_paths,
        }


def collate_sysu_ir_vcm_ir(batch):
    """Flatten SYSU image samples and VCM K-frame tracklet samples for ReID training."""
    images = []
    labels = []
    pids = []
    camids = []
    modalities = []
    tracklet_groups = []
    sources = []
    tracklet_ids = []
    paths = []
    next_group = 0

    for sample in batch:
        sample_images = sample.get("images")
        if sample_images is None:
            sample_images = sample["image"].unsqueeze(0)
        frame_paths = sample.get("frame_paths") or [sample.get("path")]
        is_vcm_tracklet = sample.get("source") == "vcm" and sample_images.shape[0] > 1
        group_id = next_group if is_vcm_tracklet else -1
        if is_vcm_tracklet:
            next_group += 1

        for frame_index in range(sample_images.shape[0]):
            images.append(sample_images[frame_index])
            labels.append(int(sample["label"]))
            pids.append(int(sample["pid"]))
            camids.append(int(sample["camid"]))
            modalities.append(int(sample["modality"]))
            tracklet_groups.append(group_id)
            sources.append(sample.get("source", "unknown"))
            tracklet_ids.append(sample.get("tracklet_id", ""))
            paths.append(frame_paths[frame_index] if frame_index < len(frame_paths) else frame_paths[-1])

    return {
        "image": torch.stack(images, dim=0),
        "label": torch.tensor(labels, dtype=torch.long),
        "pid": torch.tensor(pids, dtype=torch.long),
        "camid": torch.tensor(camids, dtype=torch.long),
        "modality": torch.tensor(modalities, dtype=torch.long),
        "tracklet_group": torch.tensor(tracklet_groups, dtype=torch.long),
        "source": sources,
        "tracklet_id": tracklet_ids,
        "path": paths,
    }


class SYSUIRVCMIRSampler(BatchSampler):
    """Source-balanced identity sampler for SYSU-MM01 IR and VCM IR samples."""

    def __init__(
        self,
        dataset,
        sysu_ir_num_ids,
        vcm_ir_num_ids,
        num_instances,
        num_batches,
        seed=42,
    ):
        self.dataset = dataset
        self.sysu_ir_num_ids = int(sysu_ir_num_ids)
        self.vcm_ir_num_ids = int(vcm_ir_num_ids)
        self.num_instances = int(num_instances)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.epoch = 0
        self.batch_size = (self.sysu_ir_num_ids + self.vcm_ir_num_ids) * self.num_instances
        if min(self.sysu_ir_num_ids, self.vcm_ir_num_ids, self.num_instances, self.num_batches) <= 0:
            raise ValueError("source id counts, num_instances, and num_batches must be positive")

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        if len(self.dataset.sysu_ir_valid_labels) < self.sysu_ir_num_ids:
            raise RuntimeError(
                "Need {} SYSU IR identities, got {}".format(
                    self.sysu_ir_num_ids,
                    len(self.dataset.sysu_ir_valid_labels),
                )
            )
        if len(self.dataset.vcm_ir_valid_labels) < self.vcm_ir_num_ids:
            raise RuntimeError(
                "Need {} VCM IR identities, got {}".format(
                    self.vcm_ir_num_ids,
                    len(self.dataset.vcm_ir_valid_labels),
                )
            )

        for _ in range(self.num_batches):
            batch = []
            for label in rng.sample(self.dataset.sysu_ir_valid_labels, self.sysu_ir_num_ids):
                batch.extend(_sample(self.dataset.sysu_ir_by_label[label], self.num_instances, rng))
            for label in rng.sample(self.dataset.vcm_ir_valid_labels, self.vcm_ir_num_ids):
                batch.extend(_sample(self.dataset.vcm_ir_by_label[label], self.num_instances, rng))
            rng.shuffle(batch)
            yield batch
