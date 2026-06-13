import json
import os
import random
from collections import Counter, defaultdict

import torch
from PIL import Image
from torch.utils.data import BatchSampler, Dataset

from project.sysumm01.datasets.sysumm01 import build_transforms


MODALITY_TO_ID = {"rgb": 0, "ir": 1}


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_path(root, path):
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


def _sample_frames(frames, count, sampling):
    if len(frames) >= count:
        if sampling == "uniform":
            if count == 1:
                return [frames[len(frames) // 2]]
            positions = [round(index * (len(frames) - 1) / float(count - 1)) for index in range(count)]
            return [frames[int(pos)] for pos in positions]
        return random.sample(frames, count)
    return [random.choice(frames) for _ in range(count)]


def _infer_modality_from_frames(default_modality, frames):
    if not frames:
        return default_modality
    first = frames[0].replace("\\", "/").lower()
    if "/rgb/" in first or "/vis/" in first or first.endswith("_vis.jpg"):
        return "rgb"
    if "/ir/" in first or "/nir/" in first or first.endswith("_nir.jpg"):
        return "ir"
    return default_modality


def _parse_vcm_frame_metadata(frame_path):
    parts = frame_path.replace("\\", "/").split("/")
    lowered = [part.lower() for part in parts]
    if "data" not in lowered:
        return None
    data_index = lowered.index("data")
    if len(parts) <= data_index + 3:
        return None
    modality = parts[data_index + 2].lower()
    camera_text = parts[data_index + 3]
    if modality not in ("rgb", "ir") or not camera_text.lower().startswith("d"):
        return None
    try:
        return {
            "pid": int(parts[data_index + 1]),
            "modality": modality,
            "camid": int(camera_text[1:]),
        }
    except ValueError:
        return None


def _filter_frames_by_metadata(frames, pid, camid, modality):
    filtered = []
    for frame_path in frames:
        metadata = _parse_vcm_frame_metadata(frame_path)
        if metadata is None:
            filtered.append(frame_path)
            continue
        if (
            metadata["pid"] == int(pid)
            and metadata["camid"] == int(camid)
            and metadata["modality"] == modality
        ):
            filtered.append(frame_path)
    return filtered


class ExternalRGBIRDataset(Dataset):
    """External RGB-IR pretraining data with image or tracklet indices.

    The dataset keeps source-local identities separate by remapping each
    source's pids into a global contiguous label space. Tracklet samples are
    flattened by collate_external_rgb_ir after sampling K frames.
    """

    def __init__(
        self,
        indices,
        image_size,
        train_augment="strong_reid",
        frames_per_tracklet=2,
        frame_sampling="random",
    ):
        if not indices:
            raise ValueError("ExternalRGBIRDataset requires at least one index")
        self.frames_per_tracklet = int(frames_per_tracklet)
        if self.frames_per_tracklet < 1:
            raise ValueError("frames_per_tracklet must be positive")
        self.frame_sampling = frame_sampling
        if self.frame_sampling not in ("random", "uniform"):
            raise ValueError("frame_sampling must be random or uniform")
        self.transform = build_transforms(image_size=image_size, training=True, augment=train_augment)

        self.samples = []
        self.source_counts = {}
        label_offset = 0
        for index_config in indices:
            name = index_config["name"]
            root = index_config.get("root")
            payload = _load_json(index_config["index"])
            root = root or payload.get("metadata", {}).get("root")
            if root is None:
                raise ValueError("Index {} has no root and no root was configured".format(index_config["index"]))
            root = os.path.abspath(root)

            raw_items = self._read_items(name, root, payload)
            pids = sorted({item["pid"] for item in raw_items})
            pid_to_label = {pid: label_offset + idx for idx, pid in enumerate(pids)}
            source_samples = []
            for item in raw_items:
                sample = dict(item)
                sample["source"] = name
                sample["root"] = root
                sample["label"] = pid_to_label[sample["pid"]]
                source_samples.append(sample)
            self.samples.extend(source_samples)
            modality_counts = Counter(sample["modality"] for sample in source_samples)
            self.source_counts[name] = {
                "samples": len(source_samples),
                "pids": len(pids),
                "rgb": int(modality_counts.get("rgb", 0)),
                "ir": int(modality_counts.get("ir", 0)),
            }
            label_offset += len(pids)

        self.num_classes = label_offset
        self.indices_by_label = defaultdict(list)
        self.rgb_by_pid = defaultdict(list)
        self.ir_by_pid = defaultdict(list)
        self.labels_by_source = defaultdict(set)
        for index, sample in enumerate(self.samples):
            label = sample["label"]
            self.indices_by_label[label].append(index)
            self.labels_by_source[sample["source"]].add(label)
            if sample["modality"] == "rgb":
                self.rgb_by_pid[label].append(index)
            elif sample["modality"] == "ir":
                self.ir_by_pid[label].append(index)
            else:
                raise ValueError("Unknown modality: {}".format(sample["modality"]))

        self.valid_labels = sorted(set(self.rgb_by_pid.keys()) & set(self.ir_by_pid.keys()))
        if not self.valid_labels:
            raise RuntimeError("ExternalRGBIRDataset has no identities with both RGB and IR samples")
        self.valid_labels_by_source = {
            source: sorted(labels & set(self.valid_labels))
            for source, labels in self.labels_by_source.items()
        }
        self.ir_labels_by_source = {}
        self.rgb_labels_by_source = {}
        for source in self.labels_by_source:
            source_indices = [
                index for index, sample in enumerate(self.samples)
                if sample["source"] == source
            ]
            self.ir_labels_by_source[source] = sorted(
                {self.samples[index]["label"] for index in source_indices if self.samples[index]["modality"] == "ir"}
            )
            self.rgb_labels_by_source[source] = sorted(
                {self.samples[index]["label"] for index in source_indices if self.samples[index]["modality"] == "rgb"}
            )

    def _read_items(self, name, root, payload):
        if "samples" in payload:
            items = []
            for raw in payload["samples"]:
                path = raw.get("path")
                if not path:
                    continue
                if not os.path.isfile(_resolve_path(root, path)):
                    continue
                modality = str(raw["modality"]).lower()
                if modality == "visible":
                    modality = "rgb"
                if modality == "nir":
                    modality = "ir"
                items.append(
                    {
                        "sample_type": "image",
                        "pid": int(raw["pid"]),
                        "camid": int(raw["camid"]),
                        "modality": modality,
                        "path": path,
                    }
                )
            return items
        if "tracklets" in payload:
            items = []
            for raw in payload["tracklets"]:
                frames = list(raw.get("frames") or raw.get("frame_paths") or [])
                frames = [frame for frame in frames if os.path.isfile(_resolve_path(root, frame))]
                modality = str(raw["modality"]).lower()
                if modality == "visible":
                    modality = "rgb"
                if modality == "nir":
                    modality = "ir"
                frames = _filter_frames_by_metadata(
                    frames,
                    pid=int(raw["pid"]),
                    camid=int(raw["camid"]),
                    modality=modality,
                )
                if not frames:
                    continue
                items.append(
                    {
                        "sample_type": "tracklet",
                        "tracklet_id": raw.get("tracklet_id", len(items)),
                        "pid": int(raw["pid"]),
                        "camid": int(raw["camid"]),
                        "modality": _infer_modality_from_frames(modality, frames),
                        "frames": frames,
                    }
                )
            return items
        raise ValueError("Unsupported external index schema for {}".format(name))

    def __len__(self):
        return len(self.samples)

    def _load_image(self, root, rel_path):
        image = Image.open(_resolve_path(root, rel_path)).convert("RGB")
        return self.transform(image)

    def __getitem__(self, index):
        sample = self.samples[index]
        if sample["sample_type"] == "tracklet":
            frames = _sample_frames(sample["frames"], self.frames_per_tracklet, self.frame_sampling)
            images = [self._load_image(sample["root"], frame) for frame in frames]
            paths = [_resolve_path(sample["root"], frame) for frame in frames]
            return {
                "image": torch.stack(images, dim=0),
                "label": sample["label"],
                "pid": sample["pid"],
                "camid": sample["camid"],
                "modality": MODALITY_TO_ID[sample["modality"]],
                "source": sample["source"],
                "path": paths,
                "sample_type": "tracklet",
                "tracklet_id": sample.get("tracklet_id"),
            }

        path = sample["path"]
        return {
            "image": self._load_image(sample["root"], path),
            "label": sample["label"],
            "pid": sample["pid"],
            "camid": sample["camid"],
            "modality": MODALITY_TO_ID[sample["modality"]],
            "source": sample["source"],
            "path": _resolve_path(sample["root"], path),
            "sample_type": "image",
        }


class ExternalRGBIRSampler(BatchSampler):
    def __init__(
        self,
        dataset,
        num_ids,
        rgb_instances,
        ir_instances,
        num_batches,
        seed=42,
        source_sampling=None,
        ir_only_num_ids_by_source=None,
    ):
        self.dataset = dataset
        self.num_ids = int(num_ids)
        self.rgb_instances = int(rgb_instances)
        self.ir_instances = int(ir_instances)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.epoch = 0
        self.source_sampling = source_sampling or {}
        self.ir_only_num_ids_by_source = ir_only_num_ids_by_source or {}
        self.source_names = sorted(dataset.valid_labels_by_source.keys())
        if not self.source_names:
            raise RuntimeError("ExternalRGBIRSampler found no valid sources")

    def __len__(self):
        return self.num_batches

    @staticmethod
    def _sample(indices, count, rng):
        if len(indices) >= count:
            return rng.sample(indices, count)
        return [rng.choice(indices) for _ in range(count)]

    def _ids_per_source(self):
        default_ratio = 1.0 / len(self.source_names) if not self.source_sampling else 0.0
        ratios = {
            source: float(self.source_sampling.get("{}_ratio".format(source), default_ratio))
            for source in self.source_names
        }
        total = sum(max(value, 0.0) for value in ratios.values())
        if total <= 0:
            return {source: 0 for source in self.source_names}
        counts = {source: int(round(self.num_ids * max(ratio, 0.0) / total)) for source, ratio in ratios.items()}
        while sum(counts.values()) < self.num_ids:
            source = max(self.source_names, key=lambda key: ratios[key] - counts[key] / float(self.num_ids))
            counts[source] += 1
        while sum(counts.values()) > self.num_ids:
            source = max(self.source_names, key=lambda key: counts[key])
            counts[source] -= 1
        return counts

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        ids_per_source = self._ids_per_source()
        fallback_labels = list(self.dataset.valid_labels)
        for _ in range(self.num_batches):
            labels = []
            for source in self.source_names:
                candidates = self.dataset.valid_labels_by_source.get(source, [])
                count = ids_per_source.get(source, 0)
                if count <= 0:
                    continue
                labels.extend(self._sample(candidates or fallback_labels, count, rng))
            if len(labels) < self.num_ids:
                labels.extend(self._sample(fallback_labels, self.num_ids - len(labels), rng))
            labels = labels[: self.num_ids]

            batch = []
            for label in labels:
                batch.extend(self._sample(self.dataset.rgb_by_pid[label], self.rgb_instances, rng))
                batch.extend(self._sample(self.dataset.ir_by_pid[label], self.ir_instances, rng))
            for source, count in self.ir_only_num_ids_by_source.items():
                count = int(count)
                if count <= 0:
                    continue
                candidates = [
                    label for label in self.dataset.ir_labels_by_source.get(source, [])
                    if self.dataset.ir_by_pid.get(label) and label not in labels
                ]
                if not candidates:
                    continue
                for label in self._sample(candidates, count, rng):
                    batch.extend(self._sample(self.dataset.ir_by_pid[label], self.ir_instances, rng))
            rng.shuffle(batch)
            yield batch


def collate_external_rgb_ir(batch):
    images = []
    labels = []
    pids = []
    camids = []
    modalities = []
    sources = []
    paths = []
    sample_types = []
    for item in batch:
        image = item["image"]
        if image.ndim == 3:
            image = image.unsqueeze(0)
        count = image.shape[0]
        images.append(image)
        labels.extend([int(item["label"])] * count)
        pids.extend([int(item["pid"])] * count)
        camids.extend([int(item["camid"])] * count)
        modalities.extend([int(item["modality"])] * count)
        sources.extend([item["source"]] * count)
        sample_types.extend([item["sample_type"]] * count)
        path = item["path"]
        if isinstance(path, list):
            paths.extend(path)
        else:
            paths.extend([path] * count)

    return {
        "image": torch.cat(images, dim=0),
        "label": torch.tensor(labels, dtype=torch.long),
        "pid": torch.tensor(pids, dtype=torch.long),
        "camid": torch.tensor(camids, dtype=torch.long),
        "modality": torch.tensor(modalities, dtype=torch.long),
        "source": sources,
        "path": paths,
        "sample_type": sample_types,
    }
