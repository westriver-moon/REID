from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_transformer import ViT


class Normalize(nn.Module):
    def __init__(self, power=2):
        super().__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        return x.div(norm + 1e-12)


def weights_init_kaiming(module):
    classname = module.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(module.weight, a=0, mode="fan_out")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(module.weight, a=0, mode="fan_in")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif classname.find("BatchNorm") != -1 and module.affine:
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def weights_init_classifier(module):
    if module.__class__.__name__.find("Linear") != -1:
        nn.init.normal_(module.weight, std=0.001)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class PMTModel(nn.Module):
    def __init__(self, config, num_classes: int):
        super().__init__()
        self.in_planes = int(config.model.embed_dim)
        self.base = ViT(
            img_size=(int(config.data.height), int(config.data.width)),
            patch_size=tuple(config.model.patch_size),
            stride_size=tuple(config.model.stride_size),
            embed_dim=int(config.model.embed_dim),
            depth=int(config.model.depth),
            num_heads=int(config.model.num_heads),
            mlp_ratio=float(config.model.mlp_ratio),
            qkv_bias=True,
            drop_rate=float(config.model.dropout),
            attn_drop_rate=float(config.model.attention_dropout),
            drop_path_rate=float(config.model.drop_path),
        )
        self.num_classes = int(num_classes)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.l2norm = Normalize(2)

    def load_imagenet_pretrained(self, path: str | Path, logger=print):
        return self.base.load_pretrained(path, logger=logger)

    def forward(self, x, return_dict: bool = False):
        features = self.base(x)
        bn_feat = self.bottleneck(features)
        if self.training:
            logits = self.classifier(bn_feat)
            if return_dict:
                return {"logits": logits, "features": features, "bn_features": bn_feat}
            return logits, features
        embeddings = F.normalize(bn_feat, p=2, dim=1)
        if return_dict:
            return {"embeddings": embeddings, "features": features, "bn_features": bn_feat}
        return embeddings


def build_pmt_model(config, num_classes: int | None = None):
    return PMTModel(config, num_classes or int(config.model.num_classes))

