from __future__ import annotations

import os

import torch
import torch.nn as nn

from .pmt_vit import ViT, resize_pos_embed, to_2tuple


def _unwrap_checkpoint(checkpoint):
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            return checkpoint["model"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def _normalize_checkpoint_key(key: str) -> str:
    if key.startswith("module."):
        key = key[len("module.") :]
    return key


def _is_skipped_key(key: str) -> bool:
    return key.startswith("head.") or key.startswith("head_dist.") or key == "dist_token" or "dist" in key


def _is_core_backbone_key(key: str) -> bool:
    return key.startswith(("patch_embed", "blocks", "norm", "cls_token", "pos_embed"))


class PMTViTVisual(nn.Module):
    def __init__(
        self,
        input_resolution=(288, 144),
        patch_size=(16, 16),
        stride_size=(12, 12),
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        drop_rate=0.03,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        output_dim=2048,
        pretrained_path=None,
    ):
        super().__init__()
        self.input_resolution = to_2tuple(input_resolution)
        self.output_dim = output_dim
        self.vit = ViT(
            img_size=self.input_resolution,
            patch_size=patch_size,
            stride_size=stride_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )
        if embed_dim == output_dim:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Linear(embed_dim, output_dim, bias=False)
            nn.init.normal_(self.projection.weight, std=embed_dim**-0.5)

        if pretrained_path:
            self.load_pretrained(pretrained_path)

    @property
    def input_dtype(self):
        return self.vit.patch_embed.proj.weight.dtype

    def forward(self, x, mode=None):
        del mode
        raw_tokens = self.vit(x, return_tokens=True)
        projected_tokens = self.projection(raw_tokens)
        return {
            "tokens": projected_tokens,
            "features": projected_tokens[:, 0],
            "raw_tokens": raw_tokens,
            "raw_features": raw_tokens[:, 0],
        }

    def load_pretrained(self, model_path, logger=print):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"PMT ImageNet checkpoint not found: {model_path}")

        checkpoint = torch.load(model_path, map_location="cpu")
        checkpoint = _unwrap_checkpoint(checkpoint)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"PMT checkpoint must resolve to a state dict, got {type(checkpoint)!r}")

        state = {}
        skipped = []
        resized_pos_embed = None
        for original_key, value in checkpoint.items():
            key = _normalize_checkpoint_key(original_key)
            if _is_skipped_key(key):
                skipped.append(key)
                continue
            if key == "pos_embed" and value.shape != self.vit.pos_embed.shape:
                resized_pos_embed = (tuple(value.shape), tuple(self.vit.pos_embed.shape))
                value = resize_pos_embed(
                    value,
                    self.vit.pos_embed,
                    self.vit.patch_embed.num_y,
                    self.vit.patch_embed.num_x,
                )
            if key == "patch_embed.proj.weight" and len(value.shape) < 4:
                out_ch, in_ch, height, width = self.vit.patch_embed.proj.weight.shape
                value = value.reshape(out_ch, in_ch, height, width)
            state[key] = value

        result = self.vit.load_state_dict(state, strict=False)
        missing_core = [key for key in result.missing_keys if _is_core_backbone_key(key)]
        unexpected_core = [key for key in result.unexpected_keys if _is_core_backbone_key(key)]

        logger(f"Loaded PMT ImageNet ViT weights from {model_path}")
        logger(f"Loaded keys: {len(state)}")
        logger(f"Missing keys: {len(result.missing_keys)}; Unexpected keys: {len(result.unexpected_keys)}")
        logger(f"Skipped classifier/distillation keys: {len(skipped)}")
        if skipped:
            logger(f"Skipped keys: {skipped}")
        if resized_pos_embed:
            logger(f"Resized pos_embed from {resized_pos_embed[0]} to {resized_pos_embed[1]}")
        else:
            logger("Resized pos_embed: not needed")

        if missing_core:
            raise RuntimeError(f"Missing PMT core backbone keys while loading ImageNet weights: {missing_core}")
        if unexpected_core:
            raise RuntimeError(f"Unexpected PMT core backbone keys while loading ImageNet weights: {unexpected_core}")

        return result
