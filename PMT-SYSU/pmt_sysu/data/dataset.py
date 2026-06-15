from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class SYSUData(Dataset):
    """PMT training dataset backed by official SYSU npy caches."""

    def __init__(
        self,
        data_dir: str | Path,
        transform_visible=None,
        transform_ir=None,
        color_index=None,
        thermal_index=None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.train_color_image = np.load(self.data_dir / "train_rgb_resized_img.npy", mmap_mode="r")
        self.train_color_label = np.load(self.data_dir / "train_rgb_resized_label.npy")
        self.train_thermal_image = np.load(self.data_dir / "train_ir_resized_img.npy", mmap_mode="r")
        self.train_thermal_label = np.load(self.data_dir / "train_ir_resized_label.npy")
        self.transform_visible = transform_visible
        self.transform_ir = transform_ir
        self.cIndex = color_index
        self.tIndex = thermal_index

    def set_indices(self, color_index, thermal_index) -> None:
        self.cIndex = np.asarray(color_index)
        self.tIndex = np.asarray(thermal_index)
        assert len(self.cIndex) == len(self.tIndex), "visible and IR index lengths differ"

    def __getitem__(self, index: int):
        assert self.cIndex is not None and self.tIndex is not None, "sampler indices are not set"
        color_idx = int(self.cIndex[index])
        thermal_idx = int(self.tIndex[index])
        img1 = self.train_color_image[color_idx]
        img2 = self.train_thermal_image[thermal_idx]
        target1 = int(self.train_color_label[color_idx])
        target2 = int(self.train_thermal_label[thermal_idx])
        if self.transform_visible is not None:
            img1 = self.transform_visible(img1)
        if self.transform_ir is not None:
            img2 = self.transform_ir(img2)
        return img1, img2, target1, target2

    def __len__(self) -> int:
        if self.cIndex is not None:
            return len(self.cIndex)
        return max(len(self.train_color_label), len(self.train_thermal_label))


class TestData(Dataset):
    def __init__(self, image_paths, labels, transform=None, img_size=(128, 256)) -> None:
        self.image_paths = list(image_paths)
        self.labels = np.asarray(labels)
        self.transform = transform
        self.img_size = img_size

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        image = image.resize((self.img_size[0], self.img_size[1]), Image.BILINEAR)
        arr = np.asarray(image)
        if self.transform is not None:
            arr = self.transform(arr)
        return arr, int(self.labels[index])

    def __len__(self) -> int:
        return len(self.image_paths)

