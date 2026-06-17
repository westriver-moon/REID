import json
import os
import random

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms as T
from torchvision.transforms import functional as TF


LIP_PART_GROUPS = {
    "head": (1, 2, 4, 13),
    "upper": (3, 5, 6, 7, 11, 14, 15),
    "lower": (9, 10, 12, 16, 17),
    "shoes": (8, 18, 19),
}


def make_horizontal_part_mask(size, num_parts=4):
    width, height = size
    mask = np.zeros((num_parts, height, width), dtype=np.float32)
    for part_index in range(num_parts):
        y0 = int(round(part_index * height / float(num_parts)))
        y1 = int(round((part_index + 1) * height / float(num_parts)))
        mask[part_index, y0:y1, :] = 1.0
    return mask


def mask_for_labels(label_mask, labels):
    part_mask = np.zeros(label_mask.shape, dtype=bool)
    for label in labels:
        part_mask |= label_mask == label
    return part_mask


def _patch_occupancy(binary, grid_h=23, grid_w=11):
    h, w = binary.shape
    values = np.zeros((grid_h, grid_w), dtype=np.float32)
    for r in range(grid_h):
        y0 = int(round(r * h / float(grid_h)))
        y1 = int(round((r + 1) * h / float(grid_h)))
        for c in range(grid_w):
            x0 = int(round(c * w / float(grid_w)))
            x1 = int(round((c + 1) * w / float(grid_w)))
            patch = binary[y0:y1, x0:x1]
            values[r, c] = float(patch.mean()) if patch.size else 0.0
    return values


def evaluate_label_mask_quality(label_mask, grid_h=23, grid_w=11):
    label_mask = np.asarray(label_mask, dtype=np.uint8)
    h, w = label_mask.shape
    image_area = float(h * w)
    foreground = label_mask > 0
    foreground_pixels = int(foreground.sum())
    foreground_ratio = foreground_pixels / max(image_area, 1.0)

    coords = np.argwhere(foreground)
    bbox_area_ratio = 0.0
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        bbox_area_ratio = ((x1 - x0) * (y1 - y0)) / max(image_area, 1.0)

    metrics = {
        "foreground_ratio": foreground_ratio,
        "bbox_area_ratio": bbox_area_ratio,
    }
    valid_parts = 0
    for name, labels in LIP_PART_GROUPS.items():
        part = mask_for_labels(label_mask, labels)
        ratio = float(part.sum()) / max(image_area, 1.0)
        patch_occ = _patch_occupancy(part, grid_h=grid_h, grid_w=grid_w)
        patch_count = int((patch_occ >= 0.25).sum())
        is_valid = ratio >= 0.005 and patch_count >= 1
        valid_parts += int(is_valid)
        metrics[name + "_ratio"] = ratio
        metrics[name + "_patch_count"] = patch_count
        metrics[name + "_valid"] = bool(is_valid)
    metrics["valid_part_count"] = valid_parts
    metrics["quality_ok"] = bool(
        0.12 <= foreground_ratio <= 0.88
        and 0.20 <= bbox_area_ratio <= 0.98
        and metrics["upper_valid"]
        and metrics["lower_valid"]
        and metrics["valid_part_count"] >= 3
    )
    metrics["quality_score"] = compute_quality_score(metrics)
    return metrics


def _triangular_band_score(value, low, high):
    if high <= low:
        return 0.0
    center = 0.5 * (low + high)
    radius = 0.5 * (high - low)
    if radius <= 0:
        return 0.0
    score = 1.0 - abs(float(value) - center) / radius
    return float(max(0.0, min(1.0, score)))


def compute_quality_score(metrics):
    fg_score = _triangular_band_score(metrics["foreground_ratio"], 0.12, 0.88)
    bbox_score = _triangular_band_score(metrics["bbox_area_ratio"], 0.20, 0.98)
    valid_part_score = min(float(metrics["valid_part_count"]) / 4.0, 1.0)
    upper_score = 1.0 if metrics.get("upper_valid", False) else 0.0
    lower_score = 1.0 if metrics.get("lower_valid", False) else 0.0
    score = (
        0.30 * fg_score
        + 0.25 * bbox_score
        + 0.20 * valid_part_score
        + 0.15 * upper_score
        + 0.10 * lower_score
    )
    return float(max(0.0, min(1.0, score)))


def make_quality_key(image_path, source_root, source_name):
    rel_path = os.path.relpath(os.path.abspath(image_path), os.path.abspath(source_root))
    rel_stem = os.path.splitext(rel_path.replace("\\", "/"))[0]
    return "{}/{}".format(source_name, rel_stem)


def make_quality_key_from_relative_path(relative_path, source_name):
    rel_stem = os.path.splitext(str(relative_path).replace("\\", "/"))[0]
    return "{}/{}".format(source_name, rel_stem)


def load_quality_index(index_path):
    if index_path is None:
        return None
    with open(index_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "entries" in payload:
        return payload["entries"]
    if isinstance(payload, dict):
        return payload
    raise ValueError("Invalid SCHP quality index: {}".format(index_path))


def lookup_quality_entry(
    quality_index,
    image_path=None,
    source_root=None,
    source_name=None,
    relative_path=None,
):
    if quality_index is None or source_name is None:
        return None
    if relative_path is not None:
        quality_key = make_quality_key_from_relative_path(relative_path, source_name)
    elif image_path is not None and source_root is not None:
        quality_key = make_quality_key(image_path, source_root, source_name)
    else:
        raise ValueError("lookup_quality_entry needs either relative_path or image_path + source_root")
    return quality_index.get(quality_key)


def labels_to_part_mask(label_mask, min_part_pixels=4, fallback_missing_parts=True):
    label_mask = np.asarray(label_mask)
    parts = []
    horizontal = make_horizontal_part_mask((label_mask.shape[1], label_mask.shape[0]), len(LIP_PART_GROUPS))
    for part_index, labels in enumerate(LIP_PART_GROUPS.values()):
        part_mask = np.zeros(label_mask.shape, dtype=bool)
        for label in labels:
            part_mask |= label_mask == label
        part_mask = part_mask.astype(np.float32)
        if fallback_missing_parts and float(part_mask.sum()) < float(min_part_pixels):
            part_mask = horizontal[part_index]
        parts.append(part_mask)
    return np.stack(parts, axis=0)


def resolve_schp_mask_path(image_path, mask_root, source_root, source_name):
    rel_path = os.path.relpath(os.path.abspath(image_path), os.path.abspath(source_root))
    rel_path = os.path.splitext(rel_path)[0] + ".png"
    return os.path.join(mask_root, source_name, rel_path)


def load_part_mask(
    image_path,
    image_size,
    mask_root=None,
    source_root=None,
    source_name=None,
    min_part_pixels=4,
    allow_fallback=True,
    quality_index=None,
):
    horizontal = make_horizontal_part_mask(image_size, len(LIP_PART_GROUPS))
    if mask_root and source_root and source_name:
        if quality_index is not None:
            quality_key = make_quality_key(image_path, source_root, source_name)
            quality_entry = quality_index.get(quality_key)
            if quality_entry is not None and not bool(quality_entry.get("quality_ok", False)):
                return horizontal
        mask_path = resolve_schp_mask_path(image_path, mask_root, source_root, source_name)
        if os.path.isfile(mask_path):
            label_mask = Image.open(mask_path)
            return labels_to_part_mask(
                np.asarray(label_mask),
                min_part_pixels=min_part_pixels,
                fallback_missing_parts=allow_fallback,
            )
        if not allow_fallback:
            raise FileNotFoundError("SCHP mask not found for {}: {}".format(image_path, mask_path))
    return horizontal


def _mask_to_pil_list(part_mask):
    return [Image.fromarray((np.clip(channel, 0.0, 1.0) * 255).astype(np.uint8)) for channel in part_mask]


def _pil_list_to_mask(mask_images):
    channels = [np.asarray(mask, dtype=np.float32) / 255.0 for mask in mask_images]
    return torch.from_numpy(np.stack(channels, axis=0)).float()


class PairedImagePartTransform:
    def __init__(
        self,
        image_size,
        training,
        augment="basic",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        schp_aug_config=None,
    ):
        self.height, self.width = image_size
        self.training = training
        self.augment = augment
        self.mean = mean
        self.std = std
        self.schp_aug_config = dict(schp_aug_config or {})
        self.schp_aug_enabled = bool(self.schp_aug_config.get("enabled", False))
        self.random_erasing = T.RandomErasing(p=0.5, value=0) if training and augment == "strong_reid" else None
        if augment not in ("basic", "strong_reid"):
            raise ValueError("Unsupported training augment: {}".format(augment))

    def _resize(self, image, masks):
        image = TF.resize(image, (self.height, self.width), interpolation=TF.InterpolationMode.BICUBIC)
        masks = [
            TF.resize(mask, (self.height, self.width), interpolation=TF.InterpolationMode.NEAREST)
            for mask in masks
        ]
        return image, masks

    @staticmethod
    def _combined_foreground_mask(masks):
        combined = np.zeros((masks[0].height, masks[0].width), dtype=np.uint8)
        for mask in masks:
            combined |= (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
        return Image.fromarray(combined, mode="L")

    @staticmethod
    def _invert_mask(mask):
        return Image.fromarray(255 - np.asarray(mask, dtype=np.uint8), mode="L")

    def _apply_background_suppress(self, image, masks):
        if not masks:
            return image
        foreground_mask = self._combined_foreground_mask(masks)
        background_mask = self._invert_mask(foreground_mask)

        blur_prob = float(self.schp_aug_config.get("blur_prob", 0.4))
        gray_prob = float(self.schp_aug_config.get("gray_prob", 0.2))
        dim_prob = float(self.schp_aug_config.get("dim_prob", 0.3))
        dim_factor = float(self.schp_aug_config.get("dim_factor", 0.35))

        if random.random() < blur_prob:
            blurred = image.filter(ImageFilter.GaussianBlur(radius=2.0))
            image = Image.composite(image, blurred, foreground_mask)
        if random.random() < gray_prob:
            gray = ImageOps.grayscale(image).convert("RGB")
            image = Image.composite(image, gray, foreground_mask)
        if random.random() < dim_prob:
            image_np = np.asarray(image, dtype=np.float32)
            bg = (np.asarray(background_mask, dtype=np.float32) / 255.0)[..., None]
            image_np = image_np * (1.0 - bg) + image_np * dim_factor * bg
            image = Image.fromarray(np.clip(image_np, 0, 255).astype(np.uint8))
        return image

    def __call__(self, image, part_mask, apply_mask_aware_aug=None):
        masks = _mask_to_pil_list(part_mask)
        image, masks = self._resize(image, masks)
        if apply_mask_aware_aug is None:
            apply_mask_aware_aug = self.schp_aug_enabled

        if self.training:
            if self.augment == "strong_reid":
                image = TF.pad(image, 10)
                masks = [TF.pad(mask, 10, fill=0) for mask in masks]
                i, j, h, w = T.RandomCrop.get_params(image, output_size=(self.height, self.width))
                image = TF.crop(image, i, j, h, w)
                masks = [TF.crop(mask, i, j, h, w) for mask in masks]
            if random.random() < 0.5:
                image = TF.hflip(image)
                masks = [TF.hflip(mask) for mask in masks]
            if apply_mask_aware_aug and self.schp_aug_enabled:
                mode = self.schp_aug_config.get("mode", "background_suppress")
                prob = float(self.schp_aug_config.get("prob", 0.5))
                if mode == "background_suppress" and random.random() < prob:
                    image = self._apply_background_suppress(image, masks)

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, self.mean, self.std)
        if self.random_erasing is not None:
            image_tensor = self.random_erasing(image_tensor)

        return image_tensor, _pil_list_to_mask(masks)
