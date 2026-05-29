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

    def pool_tokens(self, tokens, part_masks=None):
        raise NotImplementedError

    def forward(self, images, part_masks=None, return_tokens=False):
        tokens = self.vit.forward_features(images)
        pooled = self.pool_tokens(tokens, part_masks=part_masks)
        if isinstance(pooled, dict):
            outputs = pooled
        else:
            features, patch_scores = pooled
            outputs = {"features": features, "patch_scores": patch_scores}
        if return_tokens:
            outputs["tokens"] = tokens
        return outputs

    def _apply_head_drop(self, features):
        head_drop = getattr(self.vit, "head_drop", None)
        if head_drop is None:
            return features
        return head_drop(features)


class StandardViTBackbone(ViTBackboneBase):
    def pool_tokens(self, tokens, part_masks=None):
        features = self.vit.forward_head(tokens, pre_logits=True)
        patch_tokens = tokens[:, self.vit.num_prefix_tokens:]
        patch_scores = patch_tokens.norm(dim=-1)
        return features, patch_scores


class PatchMeanViTBackbone(ViTBackboneBase):
    def pool_tokens(self, tokens, part_masks=None):
        patch_tokens = tokens[:, self.vit.num_prefix_tokens:]
        patch_scores = patch_tokens.norm(dim=-1)
        features = patch_tokens.mean(dim=1)
        features = self.vit.fc_norm(features)
        features = self._apply_head_drop(features)
        return features, patch_scores


class SCHPGuidedPartPatchMeanBackbone(ViTBackboneBase):
    def __init__(
        self,
        model_name,
        checkpoint_path=None,
        drop_path_rate=0.0,
        image_size=None,
        patch_embed_config=None,
        num_parts=4,
        part_temperature=0.7,
        part_prior_bias=2.0,
        part_fusion_gamma_init=0.8,
        min_prior_pixels=1.0,
    ):
        super().__init__(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            drop_path_rate=drop_path_rate,
            image_size=image_size,
            patch_embed_config=patch_embed_config,
        )
        self.num_parts = int(num_parts)
        self.part_temperature = float(part_temperature)
        self.part_prior_bias = float(part_prior_bias)
        self.min_prior_pixels = float(min_prior_pixels)
        self.part_score_norm = nn.LayerNorm(self.feature_dim)
        self.part_score = nn.Linear(self.feature_dim, 1)
        self.part_gate = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, max(self.feature_dim // 4, 32)),
            nn.ReLU(inplace=True),
            nn.Linear(max(self.feature_dim // 4, 32), 1),
        )
        self.part_fusion_gamma = nn.Parameter(torch.tensor(float(part_fusion_gamma_init)))
        self.part_drop = nn.Dropout(p=0.0)
        nn.init.zeros_(self.part_score.weight)
        nn.init.zeros_(self.part_score.bias)

    def _horizontal_priors(self, batch_size, grid_height, grid_width, device, dtype):
        priors = torch.zeros(batch_size, self.num_parts, grid_height, grid_width, device=device, dtype=dtype)
        for part_index in range(self.num_parts):
            y0 = int(round(part_index * grid_height / float(self.num_parts)))
            y1 = int(round((part_index + 1) * grid_height / float(self.num_parts)))
            priors[:, part_index, y0:y1, :] = 1.0
        return priors

    def _part_priors(self, part_masks, batch_size, grid_height, grid_width, device, dtype):
        horizontal = self._horizontal_priors(batch_size, grid_height, grid_width, device, dtype)
        if part_masks is None:
            return horizontal
        priors = part_masks.to(device=device, dtype=dtype)
        if priors.ndim != 4:
            raise ValueError("part_masks must be [B, P, H, W], got {}".format(tuple(priors.shape)))
        if priors.shape[1] != self.num_parts:
            raise ValueError("part_masks has {} parts, expected {}".format(priors.shape[1], self.num_parts))
        priors = F.interpolate(priors, size=(grid_height, grid_width), mode="area").clamp_(0.0, 1.0)
        part_area = priors.flatten(2).sum(dim=-1, keepdim=True).unsqueeze(-1)
        missing = part_area < self.min_prior_pixels
        return torch.where(missing, horizontal, priors)

    def pool_tokens(self, tokens, part_masks=None):
        patch_tokens = tokens[:, self.vit.num_prefix_tokens:]
        batch_size, token_count, _ = patch_tokens.shape
        grid_height, grid_width = tuple(self.vit.patch_embed.grid_size)
        if grid_height * grid_width != token_count:
            raise ValueError(
                "Patch token count {} does not match grid {}x{}".format(token_count, grid_height, grid_width)
            )

        patch_scores = patch_tokens.norm(dim=-1)
        global_features = patch_tokens.mean(dim=1)
        global_features = self.vit.fc_norm(global_features)
        global_features = self._apply_head_drop(global_features)

        priors = self._part_priors(
            part_masks,
            batch_size=batch_size,
            grid_height=grid_height,
            grid_width=grid_width,
            device=patch_tokens.device,
            dtype=patch_tokens.dtype,
        ).flatten(2)

        learned_scores = self.part_score(self.part_score_norm(patch_tokens)).squeeze(-1)
        logits = learned_scores.unsqueeze(1) / max(self.part_temperature, 1e-6)
        logits = logits + self.part_prior_bias * priors
        weights = torch.softmax(logits, dim=-1)
        part_features = torch.einsum("bpn,bnc->bpc", weights, patch_tokens)
        part_features = self.vit.fc_norm(part_features)

        gates = torch.sigmoid(self.part_gate(part_features))
        fused_part = (gates * part_features).sum(dim=1) / gates.sum(dim=1).clamp_min(1e-6)
        features = global_features + self.part_fusion_gamma * self.part_drop(fused_part)
        return {
            "features": features,
            "patch_scores": patch_scores,
            "part_features": part_features,
            "part_gates": gates.squeeze(-1),
        }


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
    if model_config["type"] == "patch_mean":
        return PatchMeanViTBackbone(**common_kwargs)
    if model_config["type"] == "schp_part_patch_mean":
        part_config = model_config.get("part_aggregation", {})
        return SCHPGuidedPartPatchMeanBackbone(
            **common_kwargs,
            num_parts=part_config.get("num_parts", 4),
            part_temperature=part_config.get("temperature", 0.7),
            part_prior_bias=part_config.get("prior_bias", 2.0),
            part_fusion_gamma_init=part_config.get("fusion_gamma_init", 0.8),
            min_prior_pixels=part_config.get("min_prior_pixels", 1.0),
        )
    raise ValueError("Unknown model type: {}".format(model_config["type"]))
