import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from project.sysumm01.utils.misc import strip_prefix_if_present


_TORCHVISION_LAYER_RE = re.compile(r"^encoder\.layers\.encoder_layer_(\d+)\.(.+)$")


def _to_2tuple(value, name):
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError("{} must be an int or a length-2 sequence, got {}".format(name, value))


def _compute_grid_size(image_size, patch_size, stride):
    image_height, image_width = image_size
    patch_height, patch_width = patch_size
    stride_height, stride_width = stride
    if min(patch_height, patch_width, stride_height, stride_width) <= 0:
        raise ValueError("patch_size and stride must be positive, got {} and {}".format(patch_size, stride))
    if patch_height > image_height or patch_width > image_width:
        raise ValueError("patch_size {} exceeds image_size {}".format(patch_size, image_size))
    grid_height = (image_height - patch_height) // stride_height + 1
    grid_width = (image_width - patch_width) // stride_width + 1
    if grid_height <= 0 or grid_width <= 0:
        raise ValueError(
            "Invalid patch config for image_size {}: patch_size={}, stride={} -> grid={}".format(
                image_size,
                patch_size,
                stride,
                (grid_height, grid_width),
            )
        )
    return (grid_height, grid_width)


def _resample_patch_kernel_compat(kernel, new_size):
    new_size = tuple(new_size)
    if tuple(kernel.shape[-2:]) == new_size:
        return kernel.clone()
    out_channels, in_channels, old_height, old_width = kernel.shape
    kernel = kernel.reshape(out_channels * in_channels, 1, old_height, old_width)
    kernel = F.interpolate(kernel, size=new_size, mode="bicubic", align_corners=False)
    return kernel.reshape(out_channels, in_channels, new_size[0], new_size[1])


def _infer_old_grid_size(token_count, new_size):
    if token_count <= 0:
        raise ValueError("Invalid token count for position embedding: {}".format(token_count))

    square = int(math.sqrt(token_count))
    if square * square == token_count:
        return (square, square)

    target_ratio = float(new_size[0]) / float(new_size[1]) if new_size[1] else 1.0
    best = None
    for h in range(1, int(math.sqrt(token_count)) + 1):
        if token_count % h != 0:
            continue
        w = token_count // h
        for gh, gw in ((h, w), (w, h)):
            ratio = float(gh) / float(gw)
            score = abs(ratio - target_ratio)
            if best is None or score < best[0]:
                best = (score, gh, gw)
    if best is None:
        raise ValueError("Unable to infer position embedding grid from {} tokens".format(token_count))
    return (best[1], best[2])


def _resample_pos_embed_compat(pos_embed, new_size, num_prefix_tokens=1, old_size=None):
    prefix_tokens = pos_embed[:, :num_prefix_tokens]
    posemb_grid = pos_embed[:, num_prefix_tokens:]
    if old_size is None:
        old_size = _infer_old_grid_size(posemb_grid.shape[1], new_size)
    else:
        old_size = tuple(old_size)
        if old_size[0] * old_size[1] != posemb_grid.shape[1]:
            raise ValueError(
                "Position embedding token count {} does not match old_size {}".format(
                    posemb_grid.shape[1],
                    old_size,
                )
            )
    posemb_grid = posemb_grid.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=new_size, mode="bicubic", align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, new_size[0] * new_size[1], -1)
    return torch.cat([prefix_tokens, posemb_grid], dim=1)


def _extract_state_dict(checkpoint):
    state_dict = checkpoint
    if isinstance(state_dict, dict):
        for key in ("state_dict", "model", "model_state", "teacher", "student"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a state dict: {}".format(type(state_dict)))

    for prefix in ("module.", "model."):
        state_dict = strip_prefix_if_present(state_dict, prefix)
    return state_dict


def _looks_like_torchvision_vit(state_dict):
    return (
        "encoder.pos_embedding" in state_dict
        or "conv_proj.weight" in state_dict
        or any(key.startswith("encoder.layers.encoder_layer_") for key in state_dict)
    )


def _convert_torchvision_vit_to_timm(state_dict):
    converted = {}
    for key, value in state_dict.items():
        if key.startswith("heads.") or key == "cached_kernel":
            continue

        if key == "class_token":
            converted["cls_token"] = value
            continue
        if key == "conv_proj.weight":
            converted["patch_embed.proj.weight"] = value
            continue
        if key == "conv_proj.bias":
            converted["patch_embed.proj.bias"] = value
            continue
        if key == "encoder.pos_embedding":
            converted["pos_embed"] = value
            continue
        if key == "encoder.ln.weight":
            converted["norm.weight"] = value
            continue
        if key == "encoder.ln.bias":
            converted["norm.bias"] = value
            continue

        match = _TORCHVISION_LAYER_RE.match(key)
        if not match:
            converted[key] = value
            continue

        block_idx, subkey = match.groups()
        prefix = "blocks.{}.".format(block_idx)
        if subkey == "ln_1.weight":
            converted[prefix + "norm1.weight"] = value
        elif subkey == "ln_1.bias":
            converted[prefix + "norm1.bias"] = value
        elif subkey == "self_attention.in_proj_weight":
            converted[prefix + "attn.qkv.weight"] = value
        elif subkey == "self_attention.in_proj_bias":
            converted[prefix + "attn.qkv.bias"] = value
        elif subkey == "self_attention.out_proj.weight":
            converted[prefix + "attn.proj.weight"] = value
        elif subkey == "self_attention.out_proj.bias":
            converted[prefix + "attn.proj.bias"] = value
        elif subkey == "ln_2.weight":
            converted[prefix + "norm2.weight"] = value
        elif subkey == "ln_2.bias":
            converted[prefix + "norm2.bias"] = value
        elif subkey == "mlp.0.weight":
            converted[prefix + "mlp.fc1.weight"] = value
        elif subkey == "mlp.0.bias":
            converted[prefix + "mlp.fc1.bias"] = value
        elif subkey == "mlp.3.weight":
            converted[prefix + "mlp.fc2.weight"] = value
        elif subkey == "mlp.3.bias":
            converted[prefix + "mlp.fc2.bias"] = value
    return converted


class FusedMultiBranchPatchEmbed(nn.Module):
    def __init__(self, base_patch_embed, image_size, branches, anchor_branch=0):
        super().__init__()
        if not branches:
            raise ValueError("patch_embed.branches must contain at least one branch")

        self.img_size = tuple(image_size)
        self.in_chans = base_patch_embed.proj.in_channels
        self.embed_dim = base_patch_embed.proj.out_channels
        self.flatten = True
        self.output_fmt = getattr(base_patch_embed, "output_fmt", None)
        self.strict_img_size = getattr(base_patch_embed, "strict_img_size", True)
        self.dynamic_img_pad = getattr(base_patch_embed, "dynamic_img_pad", False)
        self.norm = base_patch_embed.norm
        self.anchor_branch = int(anchor_branch)

        self.branch_configs = []
        self.branch_grid_sizes = []
        self.proj = nn.ModuleList()
        use_bias = base_patch_embed.proj.bias is not None

        for branch in branches:
            patch_size = _to_2tuple(branch.get("patch_size", base_patch_embed.patch_size), "patch_size")
            stride = _to_2tuple(branch.get("stride", patch_size), "stride")
            self.branch_configs.append({"patch_size": patch_size, "stride": stride})
            self.branch_grid_sizes.append(_compute_grid_size(self.img_size, patch_size, stride))
            self.proj.append(
                nn.Conv2d(
                    self.in_chans,
                    self.embed_dim,
                    kernel_size=patch_size,
                    stride=stride,
                    bias=use_bias,
                )
            )

        if self.anchor_branch < 0 or self.anchor_branch >= len(self.proj):
            raise ValueError(
                "anchor_branch {} is out of range for {} branches".format(self.anchor_branch, len(self.proj))
            )

        self.patch_size = self.branch_configs[self.anchor_branch]["patch_size"]
        self.stride = self.branch_configs[self.anchor_branch]["stride"]
        self.grid_size = self.branch_grid_sizes[self.anchor_branch]
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        if len(self.proj) > 1:
            self.fuse = nn.Conv2d(self.embed_dim * len(self.proj), self.embed_dim, kernel_size=1, bias=True)
            self._init_fuse_as_anchor_identity()
        else:
            self.fuse = None

        self.initialize_from_base_patch_embed(base_patch_embed.proj.weight.detach(), base_patch_embed.proj.bias)

    def _init_fuse_as_anchor_identity(self):
        with torch.no_grad():
            self.fuse.weight.zero_()
            self.fuse.bias.zero_()
            start = self.anchor_branch * self.embed_dim
            for channel in range(self.embed_dim):
                self.fuse.weight[channel, start + channel, 0, 0] = 1.0

    def initialize_from_base_patch_embed(self, weight, bias=None):
        for branch_proj, branch_config in zip(self.proj, self.branch_configs):
            resized_weight = _resample_patch_kernel_compat(weight, branch_config["patch_size"])
            with torch.no_grad():
                branch_proj.weight.copy_(resized_weight)
                if bias is not None and branch_proj.bias is not None:
                    branch_proj.bias.copy_(bias)

    def load_from_state_dict_fragment(self, state_dict):
        weight = state_dict.pop("patch_embed.proj.weight", None)
        bias = state_dict.pop("patch_embed.proj.bias", None)
        if weight is not None:
            self.initialize_from_base_patch_embed(weight, bias)

    def forward(self, x):
        height, width = x.shape[-2:]
        if self.strict_img_size and (height, width) != self.img_size:
            raise ValueError(
                "Input image size {} does not match configured patch embed size {}".format(
                    (height, width),
                    self.img_size,
                )
            )

        feature_maps = []
        for branch_proj in self.proj:
            feature_map = branch_proj(x)
            if feature_map.shape[-2:] != self.grid_size:
                feature_map = F.interpolate(feature_map, size=self.grid_size, mode="bilinear", align_corners=False)
            feature_maps.append(feature_map)

        if self.fuse is None:
            fused = feature_maps[0]
        else:
            fused = self.fuse(torch.cat(feature_maps, dim=1))

        tokens = fused.flatten(2).transpose(1, 2)
        return self.norm(tokens)


class ViTBackboneBase(nn.Module):
    def __init__(self, model_name, checkpoint_path=None, drop_path_rate=0.0, image_size=None, patch_embed_config=None):
        super().__init__()
        create_kwargs = {
            "pretrained": False,
            "num_classes": 0,
            "drop_path_rate": drop_path_rate,
        }
        if image_size is not None:
            create_kwargs["img_size"] = tuple(image_size)
        self.vit = timm.create_model(model_name, **create_kwargs)
        if patch_embed_config:
            self._replace_patch_embed(patch_embed_config, image_size=image_size)
        self.feature_dim = getattr(self.vit, "num_features", 768)
        if checkpoint_path:
            self.load_pretrained(checkpoint_path)

    def _replace_patch_embed(self, patch_embed_config, image_size=None):
        image_size = tuple(image_size or self.vit.patch_embed.img_size)
        old_grid_size = tuple(self.vit.patch_embed.grid_size)
        patch_embed = FusedMultiBranchPatchEmbed(
            base_patch_embed=self.vit.patch_embed,
            image_size=image_size,
            branches=patch_embed_config.get("branches", []),
            anchor_branch=patch_embed_config.get("anchor_branch", 0),
        )
        self.vit.patch_embed = patch_embed
        if self.vit.pos_embed.shape[1] != self.vit.num_prefix_tokens + self.vit.patch_embed.num_patches:
            self.vit.pos_embed = nn.Parameter(
                _resample_pos_embed_compat(
                    self.vit.pos_embed.detach(),
                    new_size=tuple(self.vit.patch_embed.grid_size),
                    num_prefix_tokens=self.vit.num_prefix_tokens,
                    old_size=old_grid_size,
                )
            )

    def load_pretrained(self, checkpoint_path):
        state_dict = _extract_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        if _looks_like_torchvision_vit(state_dict):
            state_dict = _convert_torchvision_vit_to_timm(state_dict)
        if hasattr(self.vit.patch_embed, "load_from_state_dict_fragment"):
            self.vit.patch_embed.load_from_state_dict_fragment(state_dict)
        if "pos_embed" in state_dict and state_dict["pos_embed"].shape != self.vit.pos_embed.shape:
            state_dict["pos_embed"] = _resample_pos_embed_compat(
                state_dict["pos_embed"],
                new_size=tuple(self.vit.patch_embed.grid_size),
                num_prefix_tokens=self.vit.num_prefix_tokens,
            )
        msg = self.vit.load_state_dict(state_dict, strict=False)
        return msg

    def pool_tokens(self, tokens):
        raise NotImplementedError

    def forward(self, images, return_tokens=False):
        tokens = self.vit.forward_features(images)
        features, patch_scores = self.pool_tokens(tokens)
        outputs = {"features": features, "patch_scores": patch_scores}
        if return_tokens:
            outputs["tokens"] = tokens
        return outputs


class StandardViTBackbone(ViTBackboneBase):
    def pool_tokens(self, tokens):
        features = self.vit.forward_head(tokens, pre_logits=True)
        patch_tokens = tokens[:, self.vit.num_prefix_tokens:]
        patch_scores = patch_tokens.norm(dim=-1)
        return features, patch_scores


class LASTViTBackbone(ViTBackboneBase):
    def __init__(
        self,
        model_name,
        checkpoint_path=None,
        drop_path_rate=0.0,
        image_size=None,
        patch_embed_config=None,
        topk=1,
        part_token_config=None,
        eps=1e-6,
    ):
        super().__init__(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            drop_path_rate=drop_path_rate,
            image_size=image_size,
            patch_embed_config=patch_embed_config,
        )
        self.topk = topk
        self.eps = eps
        self.part_token_config = part_token_config or {}
        self.use_part_tokens = bool(self.part_token_config.get("enabled", False))
        self.num_parts = int(self.part_token_config.get("num_parts", 3))
        self.topk_per_part = int(self.part_token_config.get("topk_per_part", 4))
        self.include_global_token = bool(self.part_token_config.get("include_global", True))
        self.include_cls_token = bool(self.part_token_config.get("include_cls_token", True))
        if self.use_part_tokens and self.num_parts <= 0:
            raise ValueError("part_token_config.num_parts must be positive")

        self.row_part_indices = self._build_row_part_indices() if self.use_part_tokens else []
        fusion_token_count = 0
        if self.include_global_token:
            fusion_token_count += 1
        if self.include_cls_token:
            fusion_token_count += 1
        fusion_token_count += len(self.row_part_indices)
        self.part_token_gate = nn.Linear(self.feature_dim, 1) if self.use_part_tokens and fusion_token_count > 1 else None
        self.register_buffer("cached_kernel", None, persistent=False)

    @staticmethod
    def gaussian_kernel_1d(kernel_size, sigma, device):
        radius = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, device=device).float()
        kernel = torch.exp(-0.5 * (radius / sigma) ** 2)
        return kernel / kernel.max()

    def _get_kernel(self, dim, device):
        cached = self.cached_kernel
        if cached is None or cached.shape[-1] != dim or cached.device != device:
            kernel = self.gaussian_kernel_1d(dim, dim ** 0.5, device=device).view(1, 1, dim)
            self.cached_kernel = kernel
            return kernel
        return cached

    def _build_row_part_indices(self):
        grid_h, grid_w = tuple(self.vit.patch_embed.grid_size)
        row_splits = torch.linspace(0, grid_h, steps=self.num_parts + 1).round().to(dtype=torch.long)
        part_indices = []
        for part_idx in range(self.num_parts):
            start = int(row_splits[part_idx].item())
            end = int(row_splits[part_idx + 1].item())
            if end <= start:
                continue
            indices = []
            for row in range(start, end):
                row_offset = row * grid_w
                indices.extend(range(row_offset, row_offset + grid_w))
            if indices:
                part_indices.append(torch.tensor(indices, dtype=torch.long))
        if not part_indices:
            raise ValueError("part_token_config produced empty row partitions")
        return part_indices

    @staticmethod
    def _gather_topk_tokens(patch_tokens, scores, topk):
        topk = max(1, min(topk, scores.shape[1]))
        indices = scores.topk(k=topk, dim=1, largest=True).indices
        selected = torch.gather(
            patch_tokens,
            1,
            indices.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1]),
        )
        return selected.mean(dim=1)

    def pool_tokens(self, tokens):
        patch_tokens = tokens[:, self.vit.num_prefix_tokens:]
        kernel = self._get_kernel(patch_tokens.shape[-1], patch_tokens.device)
        spectrum = torch.fft.fft(patch_tokens, dim=-1)
        spectrum = torch.fft.fftshift(spectrum, dim=-1)
        spectrum = spectrum * kernel
        spectrum = torch.fft.ifftshift(spectrum, dim=-1)
        smoothed = torch.fft.ifft(spectrum, dim=-1).real

        diff = patch_tokens / (smoothed - patch_tokens).abs().clamp_min(self.eps)
        scores = diff.mean(dim=-1)

        global_feature = self._gather_topk_tokens(patch_tokens, scores, self.topk)
        features = global_feature
        if self.use_part_tokens:
            token_groups = []
            if self.include_global_token:
                token_groups.append(global_feature)
            if self.include_cls_token:
                token_groups.append(tokens[:, 0])

            for part_indices in self.row_part_indices:
                part_indices = part_indices.to(device=scores.device)
                part_scores = scores.index_select(1, part_indices)
                part_topk = max(1, min(self.topk_per_part, part_scores.shape[1]))
                local_topk_idx = part_scores.topk(k=part_topk, dim=1, largest=True).indices
                expanded_indices = part_indices.unsqueeze(0).expand(local_topk_idx.shape[0], -1)
                selected_absolute_idx = torch.gather(expanded_indices, 1, local_topk_idx)
                selected_tokens = torch.gather(
                    patch_tokens,
                    1,
                    selected_absolute_idx.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1]),
                )
                token_groups.append(selected_tokens.mean(dim=1))

            if len(token_groups) == 1:
                features = token_groups[0]
            else:
                stacked_tokens = torch.stack(token_groups, dim=1)
                if self.part_token_gate is None:
                    features = stacked_tokens.mean(dim=1)
                else:
                    gate_logits = self.part_token_gate(stacked_tokens).squeeze(-1)
                    gate_weights = torch.softmax(gate_logits, dim=1)
                    features = (stacked_tokens * gate_weights.unsqueeze(-1)).sum(dim=1)

        features = self.vit.fc_norm(features)
        features = self.vit.head_drop(features)
        return features, scores


def build_backbone(model_config):
    common_kwargs = {
        "model_name": model_config["backbone_name"],
        "checkpoint_path": model_config.get("pretrained_path"),
        "drop_path_rate": model_config.get("drop_path_rate", 0.0),
        "image_size": model_config.get("image_size"),
        "patch_embed_config": model_config.get("patch_embed"),
    }
    if model_config["type"] == "baseline":
        return StandardViTBackbone(**common_kwargs)
    if model_config["type"] == "lastvit":
        return LASTViTBackbone(
            topk=model_config.get("topk", 1),
            part_token_config=model_config.get("part_token"),
            **common_kwargs
        )
    raise ValueError("Unknown model type: {}".format(model_config["type"]))
