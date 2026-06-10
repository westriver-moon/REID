import torch
import torch.nn as nn
import torch.nn.functional as F

from project.sysumm01.models.backbones import build_backbone


def _extract_model_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _load_encoder_branch(backbone, bnneck, checkpoint_path, branch_name):
    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_model_state_dict(checkpoint)

    backbone_state = backbone.state_dict()
    bn_state = bnneck.state_dict()
    compatible_backbone = {}
    compatible_bn = {}
    skipped = []

    for key, value in state_dict.items():
        if key.startswith("backbone."):
            target_key = key[len("backbone.") :]
            if target_key in backbone_state and backbone_state[target_key].shape == value.shape:
                compatible_backbone[target_key] = value
            else:
                skipped.append(key)
        elif key.startswith("bnneck."):
            target_key = key[len("bnneck.") :]
            if target_key in bn_state and bn_state[target_key].shape == value.shape:
                compatible_bn[target_key] = value
            else:
                skipped.append(key)

    backbone.load_state_dict(compatible_backbone, strict=False)
    bnneck.load_state_dict(compatible_bn, strict=False)
    print(
        "Initialized {} encoder from {} (backbone_tensors={}, bnneck_tensors={}, skipped={})".format(
            branch_name,
            checkpoint_path,
            len(compatible_backbone),
            len(compatible_bn),
            len(skipped),
        ),
        flush=True,
    )


def _load_backbone_branch(backbone, checkpoint_path, branch_name):
    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_model_state_dict(checkpoint)

    backbone_state = backbone.state_dict()
    compatible_backbone = {}
    skipped = []
    for key, value in state_dict.items():
        if not key.startswith("backbone."):
            continue
        target_key = key[len("backbone.") :]
        if target_key in backbone_state and backbone_state[target_key].shape == value.shape:
            compatible_backbone[target_key] = value
        else:
            skipped.append(key)

    backbone.load_state_dict(compatible_backbone, strict=False)
    print(
        "Initialized {} frozen token encoder from {} (backbone_tensors={}, skipped={})".format(
            branch_name,
            checkpoint_path,
            len(compatible_backbone),
            len(skipped),
        ),
        flush=True,
    )


class DualEncoderReIDModel(nn.Module):
    """RGB/IR dual-encoder model with a shared embedding/classification head."""

    def __init__(self, model_config, num_classes):
        super().__init__()
        self.rgb_backbone = build_backbone(model_config["rgb_model"])
        self.ir_backbone = build_backbone(model_config["ir_model"])
        if self.rgb_backbone.feature_dim != self.ir_backbone.feature_dim:
            raise ValueError(
                "RGB and IR feature dims must match, got {} and {}".format(
                    self.rgb_backbone.feature_dim,
                    self.ir_backbone.feature_dim,
                )
            )
        feature_dim = self.rgb_backbone.feature_dim
        projector_dim = int(model_config.get("projector_dim", feature_dim))
        self.feature_dim = projector_dim

        self.rgb_bnneck = nn.BatchNorm1d(feature_dim)
        self.ir_bnneck = nn.BatchNorm1d(feature_dim)
        self.rgb_bnneck.bias.requires_grad_(False)
        self.ir_bnneck.bias.requires_grad_(False)

        adapter_config = model_config.get("adapter", {})
        self.use_adapter = bool(adapter_config.get("enabled", False))
        adapter_type = adapter_config.get("type", "residual_mlp")
        adapter_hidden_dim = int(adapter_config.get("hidden_dim", max(feature_dim // 2, 128)))
        gate_hidden_dim = int(adapter_config.get("gate_hidden_dim", max(adapter_hidden_dim // 4, 16)))
        adapter_dropout = float(adapter_config.get("dropout", 0.0) or 0.0)
        self.shared_adapter = None
        if self.use_adapter and adapter_type == "gated_shared_specific":
            self.shared_adapter = ResidualMLPDelta(feature_dim, hidden_dim=adapter_hidden_dim, dropout=adapter_dropout)
            self.rgb_adapter = GatedSharedSpecificAdapter(
                feature_dim,
                shared_adapter=self.shared_adapter,
                hidden_dim=adapter_hidden_dim,
                gate_hidden_dim=gate_hidden_dim,
                dropout=adapter_dropout,
            )
            self.ir_adapter = GatedSharedSpecificAdapter(
                feature_dim,
                shared_adapter=self.shared_adapter,
                hidden_dim=adapter_hidden_dim,
                gate_hidden_dim=gate_hidden_dim,
                dropout=adapter_dropout,
            )
        elif self.use_adapter:
            self.rgb_adapter = ModalityAdapter(feature_dim, hidden_dim=adapter_hidden_dim, dropout=adapter_dropout)
            self.ir_adapter = ModalityAdapter(feature_dim, hidden_dim=adapter_hidden_dim, dropout=adapter_dropout)
        else:
            self.rgb_adapter = nn.Identity()
            self.ir_adapter = nn.Identity()

        projector_type = model_config.get("projector", "linear")
        if projector_type == "identity":
            if projector_dim != feature_dim:
                raise ValueError("identity projector requires projector_dim == feature_dim")
            self.shared_projector = nn.Identity()
        elif projector_type == "linear":
            self.shared_projector = nn.Linear(feature_dim, projector_dim, bias=False)
            if projector_dim == feature_dim:
                nn.init.eye_(self.shared_projector.weight)
            else:
                nn.init.normal_(self.shared_projector.weight, std=0.001)
        else:
            raise ValueError("Unsupported dual encoder projector: {}".format(projector_type))
        self.classifier = nn.Linear(projector_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

        _load_encoder_branch(
            self.rgb_backbone,
            self.rgb_bnneck,
            model_config.get("rgb_init_checkpoint"),
            "RGB",
        )
        _load_encoder_branch(
            self.ir_backbone,
            self.ir_bnneck,
            model_config.get("ir_init_checkpoint"),
            "IR",
        )

    @staticmethod
    def _set_backbone_trainable(module, trainable, last_blocks=None):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        if not trainable:
            return
        if not last_blocks:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
            return
        blocks = getattr(getattr(module, "vit", None), "blocks", None)
        if blocks is None:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
            return
        for block in blocks[-int(last_blocks) :]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        for name in ("norm", "fc_norm"):
            layer = getattr(module.vit, name, None)
            if layer is not None:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)

    def set_backbones_trainable(self, trainable, last_blocks=None):
        for module in (self.rgb_backbone, self.ir_backbone):
            self._set_backbone_trainable(module, trainable, last_blocks=last_blocks)

    def set_backbones_eval(self):
        self.rgb_backbone.eval()
        self.ir_backbone.eval()

    def _forward_branch(self, images, modality_id):
        if modality_id == 0:
            outputs = self.rgb_backbone(images)
            bn_features = self.rgb_bnneck(outputs["features"])
            adapted = self.rgb_adapter(bn_features)
        else:
            outputs = self.ir_backbone(images)
            bn_features = self.ir_bnneck(outputs["features"])
            adapted = self.ir_adapter(bn_features)
        projected = self.shared_projector(adapted)
        return outputs["features"], projected, outputs.get("patch_scores")

    def forward(self, images, part_masks=None, modality=None):
        if modality is None:
            raise ValueError("DualEncoderReIDModel.forward requires modality tensor")
        if part_masks is not None:
            raise ValueError("DualEncoderReIDModel does not support part_masks")
        if not torch.is_tensor(modality):
            modality = torch.as_tensor(modality, device=images.device)
        else:
            modality = modality.to(images.device)

        batch_size = images.shape[0]
        projected = None
        global_feat = None
        patch_scores = None

        for modality_id in (0, 1):
            mask = modality == modality_id
            if not mask.any():
                continue
            branch_global, branch_projected, branch_scores = self._forward_branch(images[mask], modality_id)
            if projected is None:
                projected = branch_projected.new_zeros((batch_size, branch_projected.shape[1]))
            projected[mask] = branch_projected
            if global_feat is None:
                global_feat = branch_global.new_zeros((batch_size, branch_global.shape[1]))
            global_feat[mask] = branch_global
            if branch_scores is not None:
                if patch_scores is None:
                    patch_scores = branch_scores.new_zeros((batch_size,) + tuple(branch_scores.shape[1:]))
                patch_scores[mask] = branch_scores

        if projected is None:
            raise ValueError("DualEncoderReIDModel received no RGB/IR samples in batch")

        logits = self.classifier(projected)
        return {
            "logits": logits,
            "global_feat": projected,
            "embeddings": F.normalize(projected, dim=1),
            "patch_scores": patch_scores,
        }

    @torch.no_grad()
    def extract_features(self, images, return_patch_scores=False, part_masks=None, modality=None):
        outputs = self.forward(images, part_masks=part_masks, modality=modality)
        if return_patch_scores:
            return outputs["embeddings"], outputs["patch_scores"]
        return outputs["embeddings"]


class FrozenDualViTSharedHead(nn.Module):
    """Dual frozen ViT token encoders followed by a shared Transformer head."""

    def __init__(self, model_config, num_classes):
        super().__init__()
        self.rgb_backbone = build_backbone(model_config["rgb_model"])
        self.ir_backbone = build_backbone(model_config["ir_model"])
        if self.rgb_backbone.feature_dim != self.ir_backbone.feature_dim:
            raise ValueError(
                "RGB and IR feature dims must match, got {} and {}".format(
                    self.rgb_backbone.feature_dim,
                    self.ir_backbone.feature_dim,
                )
            )
        feature_dim = self.rgb_backbone.feature_dim
        head_config = model_config.get("shared_head", {})
        hidden_dim = int(head_config.get("hidden_dim", feature_dim))
        if hidden_dim != feature_dim:
            raise ValueError("shared_head.hidden_dim must equal backbone feature_dim for now")
        num_layers = int(head_config.get("num_layers", 2))
        num_heads = int(head_config.get("num_heads", 12))
        mlp_ratio = float(head_config.get("mlp_ratio", 4.0))
        dropout = float(head_config.get("dropout", 0.0) or 0.0)

        self.feature_dim = hidden_dim
        self.rgb_token_projector = TokenProjector(feature_dim, hidden_dim, dropout=dropout)
        self.ir_token_projector = TokenProjector(feature_dim, hidden_dim, dropout=dropout)
        self.shared_head = nn.Sequential(
            *[
                SharedTransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.head_norm = nn.LayerNorm(hidden_dim)
        self.bnneck = nn.BatchNorm1d(hidden_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(hidden_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

        _load_backbone_branch(self.rgb_backbone, model_config.get("rgb_init_checkpoint"), "RGB")
        _load_backbone_branch(self.ir_backbone, model_config.get("ir_init_checkpoint"), "IR")

    @staticmethod
    def _set_backbone_trainable(module, trainable, last_blocks=None):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        if not trainable:
            return
        if not last_blocks:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
            return
        blocks = getattr(getattr(module, "vit", None), "blocks", None)
        if blocks is None:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
            return
        for block in blocks[-int(last_blocks) :]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        for name in ("norm", "fc_norm"):
            layer = getattr(module.vit, name, None)
            if layer is not None:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)

    def set_backbones_trainable(self, trainable, last_blocks=None):
        for module in (self.rgb_backbone, self.ir_backbone):
            self._set_backbone_trainable(module, trainable, last_blocks=last_blocks)

    def set_backbones_eval(self):
        self.rgb_backbone.eval()
        self.ir_backbone.eval()

    def _forward_branch(self, images, modality_id):
        if modality_id == 0:
            outputs = self.rgb_backbone(images, return_tokens=True)
            projector = self.rgb_token_projector
            num_prefix_tokens = getattr(self.rgb_backbone.vit, "num_prefix_tokens", 1)
        else:
            outputs = self.ir_backbone(images, return_tokens=True)
            projector = self.ir_token_projector
            num_prefix_tokens = getattr(self.ir_backbone.vit, "num_prefix_tokens", 1)
        patch_tokens = outputs["tokens"][:, num_prefix_tokens:]
        projected_tokens = projector(patch_tokens)
        shared_tokens = self.shared_head(projected_tokens)
        shared_tokens = self.head_norm(shared_tokens)
        features = shared_tokens.mean(dim=1)
        patch_scores = shared_tokens.norm(dim=-1)
        return features, patch_scores

    def forward(self, images, part_masks=None, modality=None):
        if modality is None:
            raise ValueError("FrozenDualViTSharedHead.forward requires modality tensor")
        if part_masks is not None:
            raise ValueError("FrozenDualViTSharedHead does not support part_masks")
        if not torch.is_tensor(modality):
            modality = torch.as_tensor(modality, device=images.device)
        else:
            modality = modality.to(images.device)

        batch_size = images.shape[0]
        features = None
        patch_scores = None
        for modality_id in (0, 1):
            mask = modality == modality_id
            if not mask.any():
                continue
            branch_features, branch_scores = self._forward_branch(images[mask], modality_id)
            if features is None:
                features = branch_features.new_zeros((batch_size, branch_features.shape[1]))
            features[mask] = branch_features
            if patch_scores is None:
                patch_scores = branch_scores.new_zeros((batch_size,) + tuple(branch_scores.shape[1:]))
            patch_scores[mask] = branch_scores
        if features is None:
            raise ValueError("FrozenDualViTSharedHead received no RGB/IR samples in batch")

        bn_features = self.bnneck(features)
        logits = self.classifier(bn_features)
        return {
            "logits": logits,
            "global_feat": bn_features,
            "embeddings": F.normalize(bn_features, dim=1),
            "patch_scores": patch_scores,
        }

    @torch.no_grad()
    def extract_features(self, images, return_patch_scores=False, part_masks=None, modality=None):
        outputs = self.forward(images, part_masks=part_masks, modality=modality)
        if return_patch_scores:
            return outputs["embeddings"], outputs["patch_scores"]
        return outputs["embeddings"]


class ModalityAdapter(nn.Module):
    """Light residual MLP that maps modality-specific features before sharing."""

    def __init__(self, feature_dim, hidden_dim, dropout=0.0):
        super().__init__()
        layers = [
            nn.Linear(feature_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.extend(
            [
                nn.Linear(hidden_dim, feature_dim, bias=False),
                nn.BatchNorm1d(feature_dim),
            ]
        )
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, features):
        return features + self.net(features)


class ResidualMLPDelta(nn.Module):
    def __init__(self, feature_dim, hidden_dim, dropout=0.0):
        super().__init__()
        layers = [
            nn.Linear(feature_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.extend(
            [
                nn.Linear(hidden_dim, feature_dim, bias=False),
                nn.BatchNorm1d(feature_dim),
            ]
        )
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, features):
        return self.net(features)


class GatedSharedSpecificAdapter(nn.Module):
    """Fuse a shared adapter branch with a modality-specific branch."""

    def __init__(self, feature_dim, shared_adapter, hidden_dim, gate_hidden_dim, dropout=0.0):
        super().__init__()
        self.shared_adapter = shared_adapter
        self.specific_adapter = ResidualMLPDelta(feature_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, gate_hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden_dim, feature_dim),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.gate[-2].weight)
        nn.init.zeros_(self.gate[-2].bias)

    def forward(self, features):
        shared_delta = self.shared_adapter(features)
        specific_delta = self.specific_adapter(features)
        gate = self.gate(features)
        return features + gate * shared_delta + (1.0 - gate) * specific_delta


class TokenProjector(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(output_dim, output_dim),
        )
        if input_dim == output_dim:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, tokens):
        if tokens.shape[-1] == self.net[-1].out_features:
            return tokens + self.net(tokens)
        return self.net(tokens)


class SharedTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.drop1 = nn.Dropout(p=dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_hidden = int(hidden_dim * float(mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, tokens):
        normed = self.norm1(tokens)
        attn_input = normed.transpose(0, 1)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        tokens = tokens + self.drop1(attn_output.transpose(0, 1))
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens
