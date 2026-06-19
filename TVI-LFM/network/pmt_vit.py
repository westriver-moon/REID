from __future__ import annotations

import math
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ntuple(n):
    def parse(x):
        if isinstance(x, (tuple, list)):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def resize_pos_embed(posemb, posemb_new, height, width):
    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    gs_old = int(math.sqrt(len(posemb_grid)))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(height, width), mode="bilinear", align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, height * width, -1)
    return torch.cat([posemb_token, posemb_grid], dim=1)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch, tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbedOverlap(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size[0] + 1
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = self.num_x * self.num_y
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        nn.init.normal_(self.proj.weight, 0, math.sqrt(2.0 / (patch_size[0] * patch_size[1] * embed_dim)))
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        batch, channels, height, width = x.shape
        del batch, channels
        assert (height, width) == self.img_size, (
            f"Input image size ({height}*{width}) doesn't match model "
            f"({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


def _resize_patch_kernel(weight, patch_size):
    if tuple(weight.shape[-2:]) == tuple(patch_size):
        return weight
    out_ch, in_ch, _height, _width = weight.shape
    flat = weight.reshape(out_ch * in_ch, 1, weight.shape[-2], weight.shape[-1])
    resized = F.interpolate(flat, size=patch_size, mode="bicubic", align_corners=False)
    return resized.reshape(out_ch, in_ch, patch_size[0], patch_size[1])


class MultiBranchPatchEmbedOverlap(nn.Module):
    """Multi-scale overlapping patch embedding fused on the anchor token grid."""

    def __init__(self, img_size=(288, 144), branches=None, anchor_branch=0, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        branches = list(branches or [])
        if not branches:
            raise ValueError("multi-branch patch embedding requires at least one branch")
        self.img_size = img_size
        self.anchor_branch = int(anchor_branch)
        if self.anchor_branch < 0 or self.anchor_branch >= len(branches):
            raise ValueError(f"anchor_branch {self.anchor_branch} is out of range for {len(branches)} branches")

        self.branch_configs = []
        self.proj = nn.ModuleList()
        for branch in branches:
            patch_size = to_2tuple(branch.get("patch_size", 16))
            stride_size = to_2tuple(branch.get("stride_size", branch.get("stride", patch_size)))
            self.branch_configs.append({"patch_size": patch_size, "stride_size": stride_size})
            conv = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
            nn.init.normal_(conv.weight, 0, math.sqrt(2.0 / (patch_size[0] * patch_size[1] * embed_dim)))
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
            self.proj.append(conv)

        anchor = self.branch_configs[self.anchor_branch]
        self.patch_size = anchor["patch_size"]
        self.stride_size = anchor["stride_size"]
        self.num_x = (img_size[1] - self.patch_size[1]) // self.stride_size[1] + 1
        self.num_y = (img_size[0] - self.patch_size[0]) // self.stride_size[0] + 1
        self.num_patches = self.num_x * self.num_y
        self.fuse = nn.Conv2d(embed_dim * len(self.proj), embed_dim, kernel_size=1, bias=True)
        self._init_fuse_as_anchor_identity(embed_dim)

    def _init_fuse_as_anchor_identity(self, embed_dim):
        with torch.no_grad():
            self.fuse.weight.zero_()
            self.fuse.bias.zero_()
            start = self.anchor_branch * embed_dim
            for channel in range(embed_dim):
                self.fuse.weight[channel, start + channel, 0, 0] = 1.0

    def load_from_state_dict_fragment(self, state_dict):
        weight = state_dict.pop("patch_embed.proj.weight", None)
        bias = state_dict.pop("patch_embed.proj.bias", None)
        if weight is None:
            return
        if len(weight.shape) < 4:
            out_ch, in_ch = self.proj[self.anchor_branch].weight.shape[:2]
            height, width = self.proj[self.anchor_branch].weight.shape[-2:]
            weight = weight.reshape(out_ch, in_ch, height, width)
        with torch.no_grad():
            for conv, branch in zip(self.proj, self.branch_configs):
                conv.weight.copy_(_resize_patch_kernel(weight, branch["patch_size"]))
                if bias is not None and conv.bias is not None:
                    conv.bias.copy_(bias)

    def forward(self, x):
        batch, channels, height, width = x.shape
        del batch, channels
        assert (height, width) == self.img_size, (
            f"Input image size ({height}*{width}) doesn't match model "
            f"({self.img_size[0]}*{self.img_size[1]})."
        )
        target_size = (self.num_y, self.num_x)
        feature_maps = []
        for conv in self.proj:
            feature_map = conv(x)
            if feature_map.shape[-2:] != target_size:
                feature_map = F.interpolate(feature_map, size=target_size, mode="bilinear", align_corners=False)
            feature_maps.append(feature_map)
        x = self.fuse(torch.cat(feature_maps, dim=1))
        return x.flatten(2).transpose(1, 2)


class ViT(nn.Module):
    def __init__(
        self,
        img_size=(288, 144),
        patch_size=(16, 16),
        stride_size=(12, 12),
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        patch_embed_config=None,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        if patch_embed_config:
            self.patch_embed = MultiBranchPatchEmbedOverlap(
                img_size=img_size,
                branches=patch_embed_config.get("branches", []),
                anchor_branch=patch_embed_config.get("anchor_branch", 0),
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        else:
            self.patch_embed = PatchEmbedOverlap(img_size, patch_size, stride_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward_features(self, x, return_tokens=False):
        batch = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        if return_tokens:
            return x
        return x[:, 0]

    def forward(self, x, return_tokens=False):
        return self.forward_features(x, return_tokens=return_tokens)
