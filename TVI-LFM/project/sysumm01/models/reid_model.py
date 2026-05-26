import torch
import torch.nn as nn
import torch.nn.functional as F

from project.sysumm01.models.backbones import build_backbone


class ReIDModel(nn.Module):
    def __init__(self, model_config, num_classes):
        super().__init__()
        self.backbone = build_backbone(model_config)
        feature_dim = self.backbone.feature_dim
        self.bnneck = nn.BatchNorm1d(feature_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, images):
        backbone_outputs = self.backbone(images)
        global_feat = backbone_outputs["features"]
        bn_feat = self.bnneck(global_feat)
        logits = self.classifier(bn_feat)
        return {
            "logits": logits,
            "global_feat": global_feat,
            "embeddings": F.normalize(bn_feat, dim=1),
            "patch_scores": backbone_outputs["patch_scores"],
        }

    @torch.no_grad()
    def extract_features(self, images, return_patch_scores=False):
        outputs = self.forward(images)
        if return_patch_scores:
            return outputs["embeddings"], outputs["patch_scores"]
        return outputs["embeddings"]
