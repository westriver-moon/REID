import torch
import torch.nn as nn
import torch.nn.functional as F

from project.sysumm01.models.backbones import build_backbone
from project.sysumm01.models.dual_reid_model import DualEncoderReIDModel


class ReIDModel(nn.Module):
    def __init__(self, model_config, num_classes):
        super().__init__()
        self.backbone = build_backbone(model_config)
        feature_dim = self.backbone.feature_dim
        self.bnneck = nn.BatchNorm1d(feature_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
        self.part_classifier = None
        if model_config.get("part_classifier", False):
            self.part_classifier = nn.Linear(feature_dim, num_classes, bias=False)
            nn.init.normal_(self.part_classifier.weight, std=0.001)

    def forward(self, images, part_masks=None, modality=None):
        backbone_outputs = self.backbone(images, part_masks=part_masks)
        global_feat = backbone_outputs["features"]
        bn_feat = self.bnneck(global_feat)
        logits = self.classifier(bn_feat)
        outputs = {
            "logits": logits,
            "global_feat": global_feat,
            "embeddings": F.normalize(bn_feat, dim=1),
            "patch_scores": backbone_outputs["patch_scores"],
        }
        part_features = backbone_outputs.get("part_features")
        if part_features is not None:
            outputs["part_features"] = part_features
            if self.part_classifier is not None:
                part_logits = self.part_classifier(part_features.reshape(-1, part_features.shape[-1]))
                outputs["part_logits"] = part_logits.reshape(
                    part_features.shape[0],
                    part_features.shape[1],
                    -1,
                )
        return outputs

    @torch.no_grad()
    def extract_features(self, images, return_patch_scores=False, part_masks=None, modality=None):
        outputs = self.forward(images, part_masks=part_masks, modality=modality)
        if return_patch_scores:
            return outputs["embeddings"], outputs["patch_scores"]
        return outputs["embeddings"]


def build_reid_model(model_config, num_classes):
    if model_config["type"] == "dual_encoder_align":
        return DualEncoderReIDModel(model_config, num_classes)
    return ReIDModel(model_config, num_classes)
