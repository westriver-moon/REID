import os
import random

import numpy as np
import torch
from PIL import Image
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
):
    if mask_root and source_root and source_name:
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
    return make_horizontal_part_mask(image_size, len(LIP_PART_GROUPS))


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
    ):
        self.height, self.width = image_size
        self.training = training
        self.augment = augment
        self.mean = mean
        self.std = std
        self.random_erasing = T.RandomErasing(p=0.5, value=mean) if training and augment == "strong_reid" else None
        if augment not in ("basic", "strong_reid"):
            raise ValueError("Unsupported training augment: {}".format(augment))

    def _resize(self, image, masks):
        image = TF.resize(image, (self.height, self.width), interpolation=TF.InterpolationMode.BICUBIC)
        masks = [
            TF.resize(mask, (self.height, self.width), interpolation=TF.InterpolationMode.NEAREST)
            for mask in masks
        ]
        return image, masks

    def __call__(self, image, part_mask):
        masks = _mask_to_pil_list(part_mask)
        image, masks = self._resize(image, masks)

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

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, self.mean, self.std)
        if self.random_erasing is not None:
            image_tensor = self.random_erasing(image_tensor)

        return image_tensor, _pil_list_to_mask(masks)
