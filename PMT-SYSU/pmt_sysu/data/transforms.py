from __future__ import annotations

import math
import random
from typing import Callable

from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomErasing:
    """Official PMT-style random erasing after tensor normalization."""

    def __init__(
        self,
        p: float = 0.5,
        sl: float = 0.02,
        sh: float = 0.4,
        r1: float = 0.3,
        mean: tuple[float, float, float] = tuple(IMAGENET_MEAN),
    ) -> None:
        self.p = p
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) >= self.p:
            return img
        for _ in range(100):
            area = img.size(1) * img.size(2)
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.size(2) and h < img.size(1):
                x1 = random.randint(0, img.size(1) - h)
                y1 = random.randint(0, img.size(2) - w)
                for channel in range(min(img.size(0), len(self.mean))):
                    img[channel, x1 : x1 + h, y1 : y1 + w] = self.mean[channel]
                return img
        return img


class RectScale:
    def __init__(self, height: int, width: int, interpolation=Image.BILINEAR) -> None:
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        if height == self.height and width == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)


def build_transforms(height: int, width: int) -> dict[str, Callable]:
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    mix_aug = [
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.GaussianBlur(21, sigma=(0.1, 3)),
    ]
    return {
        "rgb2gray": transforms.Compose(
            [
                transforms.ToPILImage(),
                RectScale(height, width),
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                normalize,
                RandomErasing(p=0.5),
            ]
        ),
        "rgb": transforms.Compose(
            [
                transforms.ToPILImage(),
                RectScale(height, width),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                RandomErasing(p=0.5),
            ]
        ),
        "thermal": transforms.Compose(
            [
                transforms.ToPILImage(),
                RectScale(height, width),
                transforms.RandomHorizontalFlip(),
                transforms.RandomChoice(mix_aug),
                transforms.ToTensor(),
                normalize,
                RandomErasing(p=0.5),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.ToPILImage(),
                RectScale(height, width),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    }

