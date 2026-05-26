import copy
import torch
from torch import nn

from network.gem_pool import GeneralizedMeanPoolingP


def _to_2tuple(value):
    if isinstance(value, (tuple, list)):
        return int(value[0]), int(value[1])
    return int(value), int(value)


DEFAULT_LASTVIT_PATCH_EMBED = {
    "anchor_branch": 0,
    "branches": [
        {"patch_size": [16, 16], "stride": [12, 12]},
        {"patch_size": [16, 8], "stride": [12, 6]},
    ],
}


from project.sysumm01.models.backbones import build_backbone  # noqa: E402


class Dual_LASTViT(nn.Module):
    """Dual-tower LASTViT visual encoder.

    RGB and IR use independent ViT backbones. This avoids low-level weight
    sharing across modalities while keeping the same output contract used by
    the TVI-LFM training pipeline.
    """

    def __init__(
        self,
        output_dim,
        input_resolution=(288, 144),
        backbone_name="vit_base_patch16_224",
        pretrained_path=None,
        pretrained_rgb_path=None,
        pretrained_ir_path=None,
        drop_path_rate=0.1,
        topk=16,
        pooling="GEM",
        patch_embed_config=None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = _to_2tuple(input_resolution)
        self.pooling = pooling
        self.backbone_name = backbone_name

        rgb_pretrained = pretrained_rgb_path or pretrained_path
        ir_pretrained = pretrained_ir_path or pretrained_path

        model_config = {
            "type": "lastvit",
            "backbone_name": backbone_name,
            "pretrained_path": None,
            "drop_path_rate": drop_path_rate,
            "image_size": list(self.input_resolution),
            "topk": topk,
            "patch_embed": copy.deepcopy(patch_embed_config or DEFAULT_LASTVIT_PATCH_EMBED),
        }

        rgb_config = copy.deepcopy(model_config)
        rgb_config["pretrained_path"] = rgb_pretrained
        ir_config = copy.deepcopy(model_config)
        ir_config["pretrained_path"] = ir_pretrained

        self.rgb_backbone = build_backbone(rgb_config)
        self.ir_backbone = build_backbone(ir_config)

        self.feature_dim = self.rgb_backbone.feature_dim
        if self.ir_backbone.feature_dim != self.feature_dim:
            raise ValueError("RGB and IR backbone feature dims mismatch")

        self.num_prefix_tokens = self.rgb_backbone.vit.num_prefix_tokens
        self.num_y, self.num_x = tuple(self.rgb_backbone.vit.patch_embed.grid_size)
        ir_grid = tuple(self.ir_backbone.vit.patch_embed.grid_size)
        if ir_grid != (self.num_y, self.num_x):
            raise ValueError(f"RGB/IR grid mismatch: rgb={(self.num_y, self.num_x)}, ir={ir_grid}")

        self.GEM = GeneralizedMeanPoolingP()
        if pooling != "GEM":
            raise ValueError("Dual_LASTViT currently supports pooling='GEM' only")

        self.img_projection = nn.Identity()
        if output_dim != self.feature_dim:
            self.img_projection = nn.Linear(self.feature_dim, output_dim)

    @property
    def input_dtype(self):
        return next(self.rgb_backbone.vit.patch_embed.parameters()).dtype

    def _project_map(self, feature_map):
        feature_map = feature_map.permute(0, 2, 3, 1)
        feature_map = self.img_projection(feature_map)
        return feature_map.permute(0, 3, 1, 2)

    def _project_features(self, features):
        return self.img_projection(features)

    def _reshape_patch_tokens(self, patch_tokens):
        expected_tokens = self.num_y * self.num_x
        if patch_tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"Expected {expected_tokens} patch tokens, got {patch_tokens.shape[1]}"
            )
        return patch_tokens.transpose(1, 2).reshape(
            patch_tokens.shape[0],
            patch_tokens.shape[2],
            self.num_y,
            self.num_x,
        )

    @staticmethod
    def _concat_outputs(outputs):
        return {
            key: torch.cat([output[key] for output in outputs], dim=0)
            for key in outputs[0]
        }

    def _forward_single(self, x, backbone):
        outputs = backbone(x, return_tokens=True)
        patch_tokens = outputs["tokens"][:, self.num_prefix_tokens:]
        feature_map = self._reshape_patch_tokens(patch_tokens)
        return {
            "feat_map": self._project_map(feature_map),
            "features": self._project_features(outputs["features"]),
            "patch_scores": outputs["patch_scores"],
        }

    def forward(self, x, mode=None):
        x = x.type(self.input_dtype)
        if mode in (None, "rgb"):
            return self._forward_single(x, self.rgb_backbone)
        if mode == "ir":
            return self._forward_single(x, self.ir_backbone)
        if mode == "1/2":
            batch_size = x.shape[0] // 2
            rgb_outputs = self._forward_single(x[:batch_size], self.rgb_backbone)
            ir_outputs = self._forward_single(x[batch_size:], self.ir_backbone)
            return self._concat_outputs([rgb_outputs, ir_outputs])
        if mode == "1/3":
            batch_size = (2 * x.shape[0]) // 3
            rgb_outputs = self._forward_single(x[:batch_size], self.rgb_backbone)
            ir_outputs = self._forward_single(x[batch_size:], self.ir_backbone)
            return self._concat_outputs([rgb_outputs, ir_outputs])
        raise ValueError(
            f"Using model [{self.__class__.__name__}], mode must be None, 'rgb', 'ir', '1/2' or '1/3'"
        )
