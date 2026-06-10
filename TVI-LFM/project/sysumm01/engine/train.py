import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.sysumm01.datasets.sysumm01 import (
    CrossModalBatchSampler,
    FullCoverageIdentityBatchSampler,
    IdentityBatchSampler,
    MixedRGBTrainDataset,
    SYSUTrainDataset,
)
from project.sysumm01.datasets.vcm import SYSUIRVCMIRDataset, SYSUIRVCMIRSampler, collate_sysu_ir_vcm_ir
from project.sysumm01.engine.evaluator import evaluate_sysu
from project.sysumm01.losses.triplet import BatchHardTripletLoss
from project.sysumm01.models.reid_model import build_reid_model
from project.sysumm01.utils.config import dump_config, dump_json, load_config
from project.sysumm01.utils.misc import AverageMeter, append_metrics_row, count_parameters, ensure_dir, save_checkpoint, set_seed, strip_prefix_if_present


class TeeStream:
    def __init__(self, original_stream, log_handle):
        self.original_stream = original_stream
        self.log_handle = log_handle

    def write(self, data):
        written = self.original_stream.write(data)
        self.log_handle.write(data)
        self.original_stream.flush()
        self.log_handle.flush()
        return written if written is not None else len(data)

    def flush(self):
        self.original_stream.flush()
        self.log_handle.flush()

    def __getattr__(self, name):
        return getattr(self.original_stream, name)


class WarmupCosineScheduler:
    def __init__(self, optimizer, total_epochs, min_lr, warmup_epochs=0, warmup_init_lr=0.0):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.warmup_init_lr = warmup_init_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.next_epoch = 0
        self._set_lrs(self._compute_lrs(self.next_epoch))

    def _compute_lrs(self, epoch_index):
        epoch_index = max(0, epoch_index)
        if self.warmup_epochs > 0 and epoch_index < self.warmup_epochs:
            if self.warmup_epochs == 1:
                factor = 1.0
            else:
                factor = epoch_index / float(self.warmup_epochs - 1)
            return [
                self.warmup_init_lr + (base_lr - self.warmup_init_lr) * factor
                for base_lr in self.base_lrs
            ]

        cosine_span = max(1, self.total_epochs - self.warmup_epochs)
        cosine_epoch = min(max(epoch_index - self.warmup_epochs, 0), cosine_span - 1)
        if cosine_span == 1:
            progress = 1.0
        else:
            progress = cosine_epoch / float(cosine_span - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.min_lr + (base_lr - self.min_lr) * cosine for base_lr in self.base_lrs]

    def _set_lrs(self, lrs):
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr

    def step(self):
        self.next_epoch += 1
        self._set_lrs(self._compute_lrs(self.next_epoch))

    def state_dict(self):
        return {
            "total_epochs": self.total_epochs,
            "min_lr": self.min_lr,
            "warmup_epochs": self.warmup_epochs,
            "warmup_init_lr": self.warmup_init_lr,
            "base_lrs": self.base_lrs,
            "next_epoch": self.next_epoch,
        }

    def load_state_dict(self, state_dict):
        self.total_epochs = state_dict["total_epochs"]
        self.min_lr = state_dict["min_lr"]
        self.warmup_epochs = state_dict["warmup_epochs"]
        self.warmup_init_lr = state_dict["warmup_init_lr"]
        self.base_lrs = list(state_dict["base_lrs"])
        self.next_epoch = state_dict["next_epoch"]
        self._set_lrs(self._compute_lrs(self.next_epoch))


def build_scheduler(optimizer, train_config):
    scheduler_name = train_config.get("scheduler", "cosine")
    if scheduler_name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=train_config["epochs"],
            eta_min=train_config.get("min_lr", 1e-6),
        )
    if scheduler_name == "warmup_cosine":
        return WarmupCosineScheduler(
            optimizer,
            total_epochs=train_config["epochs"],
            min_lr=train_config.get("min_lr", 1e-5),
            warmup_epochs=train_config.get("warmup_epochs", 5),
            warmup_init_lr=train_config.get("warmup_init_lr", 1e-5),
        )
    raise ValueError("Unsupported scheduler: {}".format(scheduler_name))


def get_schp_eval_kwargs(config):
    eval_config = config.get("eval", {})
    return {
        "schp_mask_root": eval_config.get("schp_mask_root"),
        "schp_min_part_pixels": eval_config.get("schp_min_part_pixels", 4),
        "schp_allow_fallback": eval_config.get("schp_allow_fallback", True),
        "schp_quality_index": eval_config.get("schp_quality_index"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train the local SYSU-MM01 model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--log-file", default=None, help="Optional path to a log file; defaults to <output>/train.log")
    parser.add_argument("--print-freq", type=int, default=10, help="How often to print batch progress within each epoch")
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    return args


def parse_config_overrides(items):
    overrides = {}
    for item in items:
        if "=" not in item:
            raise ValueError("Override must be key=value, got {}".format(item))
        key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        cursor = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return overrides


def install_stream_tee(log_path):
    log_dir = os.path.dirname(log_path)
    if log_dir:
        ensure_dir(log_dir)
    log_handle = open(log_path, "a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_handle)
    sys.stderr = TeeStream(original_stderr, log_handle)
    return log_handle, original_stdout, original_stderr


def evaluate_and_save(model, config, output_dir, device, epoch, num_trials):
    eval_dir = os.path.join(output_dir, "eval")
    ensure_dir(eval_dir)
    primary_mode = config["eval"].get("primary_mode", "all")
    protocol = config["eval"].get("protocol", "cross_modality")
    modality = config["eval"].get("modality")
    print(
        "[Epoch {:03d}] evaluating on SYSU-MM01 (mode={}, protocol={}, modality={}, trials={})".format(
            epoch,
            primary_mode,
            protocol,
            modality or "cross",
            num_trials,
        ),
        flush=True,
    )
    all_metrics, all_retrievals = evaluate_sysu(
        model=model,
        dataset_root=config["eval"].get("dataset_root", config["dataset"].get("root")),
        image_size=tuple(config["dataset"]["image_size"]),
        batch_size=config["eval"]["batch_size"],
        num_workers=config["eval"]["num_workers"],
        device=device,
        mode=primary_mode,
        num_trials=num_trials,
        seed=config["seed"] + epoch,
        protocol=protocol,
        modality=modality,
        **get_schp_eval_kwargs(config),
    )
    payload = {"metrics": all_metrics, "retrieval_examples": all_retrievals, "epoch": epoch}
    dump_json(payload, os.path.join(eval_dir, "{}_epoch_{:03d}.json".format(primary_mode, epoch)))
    print(
        "[Epoch {:03d}] evaluation done: mAP={:.4f}, Rank-1={:.4f}".format(
            epoch,
            all_metrics["mAP"],
            all_metrics["rank1"],
        ),
        flush=True,
    )
    return all_metrics


def _extract_model_state_dict(checkpoint):
    state_dict = checkpoint
    if isinstance(state_dict, dict):
        for key in ("model", "state_dict", "model_state"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a model state dict: {}".format(type(state_dict)))
    for prefix in ("module.",):
        state_dict = strip_prefix_if_present(state_dict, prefix)
    return state_dict


def initialize_model_weights(model, checkpoint_path, init_config=None):
    init_config = init_config or {}
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_model_state_dict(checkpoint)
    model_state = model.state_dict()
    compatible_state = {}
    skipped = []
    partial_loaded = []
    partial_classifier = init_config.get("init_partial_classifier", False)
    partial_rows = init_config.get("partial_classifier_rows")
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible_state[key] = value
        elif (
            partial_classifier
            and key == "classifier.weight"
            and key in model_state
            and value.ndim == 2
            and model_state[key].ndim == 2
            and value.shape[1] == model_state[key].shape[1]
        ):
            rows = int(partial_rows or model_state[key].shape[0])
            rows = min(rows, value.shape[0], model_state[key].shape[0])
            merged = model_state[key].clone()
            merged[:rows] = value[:rows]
            compatible_state[key] = merged
            partial_loaded.append("{}[:{}]".format(key, rows))
        else:
            skipped.append(key)
    if partial_loaded:
        print(
            "Partially loaded checkpoint tensors: {}".format(", ".join(partial_loaded)),
            flush=True,
        )
    if skipped:
        print(
            "Skipped {} checkpoint tensors with missing keys or shape mismatch: {}".format(
                len(skipped),
                ", ".join(skipped[:12]) + (" ..." if len(skipped) > 12 else ""),
            ),
            flush=True,
        )
    msg = model.load_state_dict(compatible_state, strict=False)
    return msg


def set_backbone_trainable(model, trainable, last_blocks=None):
    if hasattr(model, "set_backbones_trainable"):
        model.set_backbones_trainable(trainable, last_blocks=last_blocks)
        return
    if not hasattr(model, "backbone"):
        return
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    if not trainable:
        return
    if not last_blocks:
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(True)
        return
    blocks = getattr(getattr(model.backbone, "vit", None), "blocks", None)
    if blocks is None:
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(True)
        return
    for block in blocks[-int(last_blocks) :]:
        for parameter in block.parameters():
            parameter.requires_grad_(True)
    for name in ("norm", "fc_norm"):
        layer = getattr(model.backbone.vit, name, None)
        if layer is not None:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)


def set_backbone_eval(model):
    if hasattr(model, "set_backbones_eval"):
        model.set_backbones_eval()
    elif hasattr(model, "backbone"):
        model.backbone.eval()


def tracklet_consistency_loss(embeddings, tracklet_groups):
    valid_groups = tracklet_groups[tracklet_groups >= 0].unique(sorted=True)
    if valid_groups.numel() == 0:
        return embeddings.new_tensor(0.0)

    normalized = nn.functional.normalize(embeddings, dim=1)
    losses = []
    for group_id in valid_groups.tolist():
        mask = tracklet_groups == group_id
        if int(mask.sum().item()) < 2:
            continue
        group_features = normalized[mask]
        center = nn.functional.normalize(group_features.mean(dim=0, keepdim=True), dim=1)
        losses.append(1.0 - (group_features * center).sum(dim=1).mean())
    if not losses:
        return embeddings.new_tensor(0.0)
    return torch.stack(losses).mean()


def _source_mask(sources, source_name, device):
    return torch.tensor([source == source_name for source in sources], device=device, dtype=torch.bool)


def _masked_ce_loss(logits, labels, mask, ce_criterion):
    if mask is None:
        return ce_criterion(logits, labels)
    if not mask.any():
        return logits.new_tensor(0.0)
    return ce_criterion(logits[mask], labels[mask])


def _masked_triplet_loss(features, labels, mask, triplet_criterion):
    if mask is None:
        return triplet_criterion(features, labels)
    if not mask.any():
        return features.new_tensor(0.0)
    return triplet_criterion(features[mask], labels[mask])


def compute_reid_losses(outputs, labels, batch, ce_criterion, triplet_criterion, loss_config):
    if loss_config.get("source_aware", False) and "source" in batch:
        sources = batch["source"]
        device = labels.device
        sysu_mask = _source_mask(sources, "sysumm01_ir", device)
        vcm_mask = _source_mask(sources, "vcm", device)

        sysu_id = _masked_ce_loss(outputs["logits"], labels, sysu_mask, ce_criterion)
        vcm_id = _masked_ce_loss(outputs["logits"], labels, vcm_mask, ce_criterion)
        sysu_tri = _masked_triplet_loss(outputs["global_feat"], labels, sysu_mask, triplet_criterion)
        vcm_tri = _masked_triplet_loss(outputs["global_feat"], labels, vcm_mask, triplet_criterion)

        id_loss = (
            loss_config.get("sysu_id_weight", 1.0) * sysu_id
            + loss_config.get("vcm_id_weight", 1.0) * vcm_id
        )
        tri_loss = (
            loss_config.get("sysu_triplet_weight", 1.0) * sysu_tri
            + loss_config.get("vcm_triplet_weight", 1.0) * vcm_tri
        )
        return id_loss, tri_loss

    id_loss = ce_criterion(outputs["logits"], labels)
    tri_loss = triplet_criterion(outputs["global_feat"], labels)
    return loss_config.get("id_weight", 1.0) * id_loss, loss_config.get("triplet_weight", 1.0) * tri_loss


def compute_part_id_loss(outputs, labels, ce_criterion, loss_config):
    weight = float(loss_config.get("part_id_weight", 0.0) or 0.0)
    if weight <= 0.0 or "part_logits" not in outputs:
        return outputs["global_feat"].new_tensor(0.0)
    part_logits = outputs["part_logits"]
    if part_logits.ndim != 3:
        raise ValueError("part_logits must be [B, P, num_classes], got {}".format(tuple(part_logits.shape)))
    part_count = part_logits.shape[1]
    flat_logits = part_logits.reshape(part_logits.shape[0] * part_count, part_logits.shape[2])
    flat_labels = labels.unsqueeze(1).expand(-1, part_count).reshape(-1)
    return weight * ce_criterion(flat_logits, flat_labels)


def cross_modal_contrast_loss(embeddings, labels, modalities, temperature=0.07):
    rgb_mask = modalities == 0
    ir_mask = modalities == 1
    if not rgb_mask.any() or not ir_mask.any():
        return embeddings.new_tensor(0.0)

    rgb_features = nn.functional.normalize(embeddings[rgb_mask], dim=1)
    ir_features = nn.functional.normalize(embeddings[ir_mask], dim=1)
    rgb_labels = labels[rgb_mask]
    ir_labels = labels[ir_mask]

    positive = rgb_labels[:, None].eq(ir_labels[None, :])
    if not positive.any():
        return embeddings.new_tensor(0.0)

    temperature = max(float(temperature), 1e-6)
    logits = rgb_features @ ir_features.t() / temperature

    def _multi_positive_nce(scores, pos_mask):
        valid = pos_mask.any(dim=1)
        if not valid.any():
            return scores.new_tensor(0.0)
        scores = scores[valid]
        pos_mask = pos_mask[valid]
        pos_scores = scores.masked_fill(~pos_mask, -torch.finfo(scores.dtype).max)
        return -(torch.logsumexp(pos_scores, dim=1) - torch.logsumexp(scores, dim=1)).mean()

    return 0.5 * (_multi_positive_nce(logits, positive) + _multi_positive_nce(logits.t(), positive.t()))


def cross_modal_proto_align_loss(embeddings, labels, modalities):
    rgb_mask = modalities == 0
    ir_mask = modalities == 1
    rgb_labels = set(labels[rgb_mask].detach().cpu().tolist())
    ir_labels = set(labels[ir_mask].detach().cpu().tolist())
    common_labels = sorted(rgb_labels & ir_labels)
    if not common_labels:
        return embeddings.new_tensor(0.0)

    normalized = nn.functional.normalize(embeddings, dim=1)
    losses = []
    for label in common_labels:
        label_mask = labels == label
        rgb_center = normalized[label_mask & rgb_mask].mean(dim=0, keepdim=True)
        ir_center = normalized[label_mask & ir_mask].mean(dim=0, keepdim=True)
        rgb_center = nn.functional.normalize(rgb_center, dim=1)
        ir_center = nn.functional.normalize(ir_center, dim=1)
        losses.append(1.0 - (rgb_center * ir_center).sum())
    return torch.stack(losses).mean()


def cross_modal_batch_hard_triplet_loss(embeddings, labels, modalities, margin=0.3):
    rgb_mask = modalities == 0
    ir_mask = modalities == 1
    if not rgb_mask.any() or not ir_mask.any():
        return embeddings.new_tensor(0.0)

    normalized = F.normalize(embeddings, dim=1)
    rgb_features = normalized[rgb_mask]
    ir_features = normalized[ir_mask]
    rgb_labels = labels[rgb_mask]
    ir_labels = labels[ir_mask]
    distances = torch.cdist(rgb_features, ir_features, p=2)
    positive = rgb_labels[:, None].eq(ir_labels[None, :])
    negative = ~positive
    if not positive.any() or not negative.any():
        return embeddings.new_tensor(0.0)

    losses = []
    rgb_valid = positive.any(dim=1) & negative.any(dim=1)
    if rgb_valid.any():
        pos_dist = distances.masked_fill(~positive, float("-inf")).max(dim=1)[0]
        neg_dist = distances.masked_fill(~negative, float("inf")).min(dim=1)[0]
        losses.append(F.relu(pos_dist[rgb_valid] - neg_dist[rgb_valid] + margin).mean())

    ir_valid = positive.any(dim=0) & negative.any(dim=0)
    if ir_valid.any():
        pos_dist = distances.masked_fill(~positive, float("-inf")).max(dim=0)[0]
        neg_dist = distances.masked_fill(~negative, float("inf")).min(dim=0)[0]
        losses.append(F.relu(pos_dist[ir_valid] - neg_dist[ir_valid] + margin).mean())

    if not losses:
        return embeddings.new_tensor(0.0)
    return torch.stack(losses).mean()


def cross_modal_circle_loss(embeddings, labels, modalities, margin=0.25, gamma=32.0):
    rgb_mask = modalities == 0
    ir_mask = modalities == 1
    if not rgb_mask.any() or not ir_mask.any():
        return embeddings.new_tensor(0.0)

    normalized = F.normalize(embeddings, dim=1)
    rgb_features = normalized[rgb_mask]
    ir_features = normalized[ir_mask]
    rgb_labels = labels[rgb_mask]
    ir_labels = labels[ir_mask]
    sim = rgb_features @ ir_features.t()
    positive = rgb_labels[:, None].eq(ir_labels[None, :])
    negative = ~positive
    if not positive.any() or not negative.any():
        return embeddings.new_tensor(0.0)

    margin = float(margin)
    gamma = float(gamma)
    op = 1.0 + margin
    on = -margin
    delta_p = 1.0 - margin
    delta_n = margin

    def _directional_loss(scores, pos_mask):
        neg_mask = ~pos_mask
        valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
        if not valid.any():
            return scores.new_tensor(0.0)
        scores = scores[valid]
        pos_mask = pos_mask[valid]
        neg_mask = ~pos_mask
        sp = scores.unsqueeze(1)
        sn = scores.unsqueeze(1)
        alpha_p = torch.clamp_min(op - sp.detach(), 0.0)
        alpha_n = torch.clamp_min(sn.detach() - on, 0.0)
        logit_p = (-gamma * alpha_p * (sp - delta_p)).squeeze(1)
        logit_n = (gamma * alpha_n * (sn - delta_n)).squeeze(1)
        logit_p = logit_p.masked_fill(~pos_mask, -torch.finfo(scores.dtype).max)
        logit_n = logit_n.masked_fill(~neg_mask, -torch.finfo(scores.dtype).max)
        return F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

    return 0.5 * (_directional_loss(sim, positive) + _directional_loss(sim.t(), positive.t()))


def cross_modal_distance_stats(embeddings, labels, modalities):
    rgb_mask = modalities == 0
    ir_mask = modalities == 1
    zero = embeddings.new_tensor(0.0)
    if not rgb_mask.any() or not ir_mask.any():
        return zero, zero, zero

    normalized = F.normalize(embeddings.detach(), dim=1)
    rgb_features = normalized[rgb_mask]
    ir_features = normalized[ir_mask]
    rgb_labels = labels[rgb_mask]
    ir_labels = labels[ir_mask]
    distances = torch.cdist(rgb_features, ir_features, p=2)
    positive = rgb_labels[:, None].eq(ir_labels[None, :])
    negative = ~positive
    if not positive.any() or not negative.any():
        return zero, zero, zero
    pos_dist = distances[positive].mean()
    neg_dist = distances[negative].mean()
    return pos_dist, neg_dist, neg_dist - pos_dist


class CrossModalPrototypeMemory:
    def __init__(self, num_classes, feature_dim, momentum=0.9):
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.momentum = float(momentum)
        self.rgb_memory = None
        self.ir_memory = None
        self.rgb_valid = None
        self.ir_valid = None

    def to(self, device, dtype=torch.float32):
        if self.rgb_memory is None:
            self.rgb_memory = torch.zeros(self.num_classes, self.feature_dim, device=device, dtype=dtype)
            self.ir_memory = torch.zeros(self.num_classes, self.feature_dim, device=device, dtype=dtype)
            self.rgb_valid = torch.zeros(self.num_classes, device=device, dtype=torch.bool)
            self.ir_valid = torch.zeros(self.num_classes, device=device, dtype=torch.bool)
        else:
            self.rgb_memory = self.rgb_memory.to(device=device, dtype=dtype)
            self.ir_memory = self.ir_memory.to(device=device, dtype=dtype)
            self.rgb_valid = self.rgb_valid.to(device=device)
            self.ir_valid = self.ir_valid.to(device=device)
        return self

    def state_dict(self):
        if self.rgb_memory is None:
            return {}
        return {
            "rgb_memory": self.rgb_memory,
            "ir_memory": self.ir_memory,
            "rgb_valid": self.rgb_valid,
            "ir_valid": self.ir_valid,
        }

    def load_state_dict(self, state_dict):
        if not state_dict:
            return
        self.rgb_memory = state_dict["rgb_memory"]
        self.ir_memory = state_dict["ir_memory"]
        self.rgb_valid = state_dict["rgb_valid"]
        self.ir_valid = state_dict["ir_valid"]

    def _batch_centers(self, embeddings, labels, mask):
        centers = {}
        if not mask.any():
            return centers
        selected_labels = labels[mask].detach()
        selected_embeddings = embeddings[mask]
        for label in selected_labels.unique(sorted=True).tolist():
            label_mask = selected_labels == label
            center = selected_embeddings[label_mask].mean(dim=0)
            centers[int(label)] = F.normalize(center.float(), dim=0)
        return centers

    def loss_and_update(self, embeddings, labels, modalities):
        self.to(embeddings.device, dtype=torch.float32)
        normalized = F.normalize(embeddings.float(), dim=1)
        rgb_centers = self._batch_centers(normalized, labels, modalities == 0)
        ir_centers = self._batch_centers(normalized, labels, modalities == 1)

        losses = []
        for label, center in rgb_centers.items():
            if self.ir_valid[label]:
                losses.append(1.0 - (center * self.ir_memory[label].detach().clone()).sum())
        for label, center in ir_centers.items():
            if self.rgb_valid[label]:
                losses.append(1.0 - (center * self.rgb_memory[label].detach().clone()).sum())

        common_labels = sorted(set(rgb_centers.keys()) & set(ir_centers.keys()))
        for label in common_labels:
            losses.append(1.0 - (rgb_centers[label] * ir_centers[label]).sum())

        with torch.no_grad():
            for label, center in rgb_centers.items():
                if self.rgb_valid[label]:
                    self.rgb_memory[label].mul_(self.momentum).add_(center.detach(), alpha=1.0 - self.momentum)
                    self.rgb_memory[label] = F.normalize(self.rgb_memory[label], dim=0)
                else:
                    self.rgb_memory[label].copy_(center.detach())
                    self.rgb_valid[label] = True
            for label, center in ir_centers.items():
                if self.ir_valid[label]:
                    self.ir_memory[label].mul_(self.momentum).add_(center.detach(), alpha=1.0 - self.momentum)
                    self.ir_memory[label] = F.normalize(self.ir_memory[label], dim=0)
                else:
                    self.ir_memory[label].copy_(center.detach())
                    self.ir_valid[label] = True

        if not losses:
            return embeddings.new_tensor(0.0)
        return torch.stack(losses).mean().to(dtype=embeddings.dtype)


def compute_strong_cross_modal_losses(outputs, labels, batch, loss_config, epoch, proto_memory=None):
    strong_config = loss_config.get("cross_modal_metric", {})
    enabled = bool(strong_config.get("enabled", False))
    start_epoch = int(strong_config.get("start_epoch", 1) or 1)
    zero = outputs["global_feat"].new_tensor(0.0)
    if not enabled or epoch < start_epoch or "modality" not in batch:
        return zero, zero, zero, zero, zero, zero

    modalities = batch["modality"]
    if not torch.is_tensor(modalities):
        modalities = torch.as_tensor(modalities, device=labels.device)
    else:
        modalities = modalities.to(labels.device, non_blocking=True)

    embeddings = outputs["embeddings"]
    cm_contrast = zero
    cm_triplet = zero
    memory_proto = zero
    cm_circle = zero

    contrast_config = strong_config.get("contrast", {})
    if bool(contrast_config.get("enabled", False)):
        cm_contrast = float(contrast_config.get("weight", 0.0) or 0.0) * cross_modal_contrast_loss(
            embeddings,
            labels,
            modalities,
            temperature=contrast_config.get("temperature", strong_config.get("temperature", 0.07)),
        )

    triplet_config = strong_config.get("triplet", {})
    if bool(triplet_config.get("enabled", False)):
        cm_triplet = float(triplet_config.get("weight", 0.0) or 0.0) * cross_modal_batch_hard_triplet_loss(
            embeddings,
            labels,
            modalities,
            margin=triplet_config.get("margin", loss_config.get("triplet_margin", 0.3)),
        )

    circle_config = strong_config.get("circle_loss", strong_config.get("cross_modal_circle_loss", {}))
    if bool(circle_config.get("enabled", False)):
        cm_circle = float(circle_config.get("weight", 0.0) or 0.0) * cross_modal_circle_loss(
            embeddings,
            labels,
            modalities,
            margin=circle_config.get("margin", 0.25),
            gamma=circle_config.get("gamma", 32.0),
        )

    memory_config = strong_config.get("prototype_memory", {})
    if proto_memory is not None and bool(memory_config.get("enabled", False)):
        memory_proto = float(memory_config.get("weight", 0.0) or 0.0) * proto_memory.loss_and_update(
            embeddings,
            labels,
            modalities,
        )

    pos_dist, neg_dist, gap = cross_modal_distance_stats(embeddings, labels, modalities)
    return cm_contrast, cm_triplet + cm_circle, memory_proto, pos_dist, neg_dist, gap


def compute_rgb_ir_alignment_losses(outputs, labels, batch, loss_config, epoch):
    align_config = loss_config.get("rgb_ir_alignment", {})
    enabled = bool(align_config.get("enabled", False))
    start_epoch = int(align_config.get("start_epoch", 1) or 1)
    zero = outputs["global_feat"].new_tensor(0.0)
    if not enabled or epoch < start_epoch or "modality" not in batch:
        return zero, zero

    modalities = batch["modality"]
    if not torch.is_tensor(modalities):
        modalities = torch.as_tensor(modalities, device=labels.device)
    else:
        modalities = modalities.to(labels.device, non_blocking=True)

    clip_weight = float(align_config.get("clip_loss_weight", 0.0) or 0.0)
    proto_weight = float(align_config.get("proto_loss_weight", 0.0) or 0.0)
    if clip_weight <= 0.0 and proto_weight <= 0.0:
        return zero, zero

    embeddings = outputs["embeddings"]
    clip_loss = zero
    proto_loss = zero
    if clip_weight > 0.0:
        clip_loss = clip_weight * cross_modal_contrast_loss(
            embeddings,
            labels,
            modalities,
            temperature=align_config.get("temperature", 0.07),
        )
    if proto_weight > 0.0:
        proto_loss = proto_weight * cross_modal_proto_align_loss(embeddings, labels, modalities)
    return clip_loss, proto_loss


def get_final_eval_modes(config):
    primary_mode = config["eval"].get("primary_mode", "all")
    modes = [primary_mode]
    for mode in config["eval"].get("extra_modes", []):
        if mode not in modes:
            modes.append(mode)
    return modes


def update_topk_checkpoints(top_records, state, metrics, output_dir, top_k):
    if top_k <= 0:
        return top_records

    epoch = int(state["epoch"])
    score = float(metrics["mAP"])
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    filename = "top_epoch_{:03d}_map_{:.6f}.pth".format(epoch, score)
    path = os.path.join(checkpoint_dir, filename)
    save_checkpoint(state, path)

    top_records = list(top_records) + [{"epoch": epoch, "mAP": score, "path": path}]
    top_records.sort(key=lambda item: item["mAP"], reverse=True)
    keep = top_records[:top_k]
    drop = top_records[top_k:]
    keep_paths = {item["path"] for item in keep}
    for item in drop:
        stale_path = item["path"]
        if stale_path not in keep_paths and os.path.exists(stale_path):
            os.remove(stale_path)
    return keep


def evaluate_topk_checkpoints(model, config, output_dir, device, top_records):
    if not top_records:
        return
    eval_dir = os.path.join(output_dir, "topk_final_eval")
    ensure_dir(eval_dir)
    summary = []
    for item in sorted(top_records, key=lambda record: record["mAP"], reverse=True):
        checkpoint = torch.load(item["path"], map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        epoch_payload = {
            "checkpoint": item["path"],
            "epoch": item["epoch"],
            "selection_mAP": item["mAP"],
            "metrics": {},
        }
        for eval_mode in get_final_eval_modes(config):
            metrics, retrievals = evaluate_sysu(
                model=model,
                dataset_root=config["eval"].get("dataset_root", config["dataset"].get("root")),
                image_size=tuple(config["dataset"]["image_size"]),
                batch_size=config["eval"]["batch_size"],
                num_workers=config["eval"]["num_workers"],
                device=device,
                mode=eval_mode,
                num_trials=config["eval"]["num_trials"],
                seed=config["seed"],
                protocol=config["eval"].get("protocol", "cross_modality"),
                modality=config["eval"].get("modality"),
                **get_schp_eval_kwargs(config),
            )
            epoch_payload["metrics"][eval_mode] = metrics
            dump_json(
                {
                    "checkpoint": item["path"],
                    "selection_mAP": item["mAP"],
                    "metrics": metrics,
                    "retrieval_examples": retrievals,
                },
                os.path.join(eval_dir, "top_epoch_{:03d}_{}_final.json".format(item["epoch"], eval_mode)),
            )
        summary.append(epoch_payload)
    dump_json(summary, os.path.join(eval_dir, "summary.json"))


def main():
    args = parse_args()
    config = load_config(args.config, overrides=parse_config_overrides(args.overrides))
    if args.seed is not None:
        config["seed"] = args.seed
    config["model"]["image_size"] = list(config["dataset"]["image_size"])

    output_dir = args.output
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, "checkpoints"))
    dump_config(config, os.path.join(output_dir, "config.yaml"))

    log_path = args.log_file or os.path.join(output_dir, "train.log")
    log_handle, original_stdout, original_stderr = install_stream_tee(log_path)
    try:
        print("Logging to {}".format(log_path), flush=True)

        set_seed(config["seed"])
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        dataset_name = config["dataset"].get("name", "sysumm01")
        if dataset_name == "mixed_rgb":
            train_dataset = MixedRGBTrainDataset(
                sysu_root=config["dataset"]["sysu_root"],
                msmt_root=config["dataset"]["msmt_root"],
                image_size=tuple(config["dataset"]["image_size"]),
                sysu_use_val=config["dataset"].get("sysu_use_val", True),
                msmt_use_val=config["dataset"].get("msmt_use_val", True),
                train_augment=config["dataset"].get("train_augment", "basic"),
            )
            config["dataset"]["resolved_roots"] = train_dataset.resolved_roots
        elif dataset_name == "sysumm01":
            train_dataset = SYSUTrainDataset(
                root=config["dataset"]["root"],
                image_size=tuple(config["dataset"]["image_size"]),
                use_val=config["dataset"].get("use_val", True),
                train_augment=config["dataset"].get("train_augment", "basic"),
                train_modality=config["dataset"].get("train_modality", "both"),
            )
        elif dataset_name == "sysu_ir_vcm_ir":
            train_dataset = SYSUIRVCMIRDataset(
                sysu_root=config["dataset"]["sysu_root"],
                vcm_root=config["dataset"]["vcm_root"],
                vcm_tracklet_json=config["dataset"]["vcm_tracklet_json"],
                image_size=tuple(config["dataset"]["image_size"]),
                sysu_use_val=config["dataset"].get("sysu_use_val", True),
                vcm_frame_sampling=config["dataset"].get("vcm_frame_sampling", "random"),
                vcm_frames_per_tracklet=config["dataset"].get("vcm_frames_per_tracklet", 1),
                train_augment=config["dataset"].get("train_augment", "strong_reid"),
                schp_mask_root=config["dataset"].get("schp_mask_root"),
                schp_min_part_pixels=config["dataset"].get("schp_min_part_pixels", 4),
                schp_allow_fallback=config["dataset"].get("schp_allow_fallback", True),
                schp_quality_index=config["dataset"].get("schp_quality_index"),
                schp_aug=config["dataset"].get("schp_aug"),
                return_part_masks=config["model"]["type"] == "schp_part_patch_mean",
                vcm_sampling_top_ratio=config["dataset"].get("vcm_sampling_top_ratio", 0.5),
                vcm_sampling_temperature=config["dataset"].get("vcm_sampling_temperature", 0.7),
            )
        else:
            raise ValueError("Unsupported dataset.name: {}".format(dataset_name))
        config["dataset"]["num_classes"] = train_dataset.num_classes
        dump_config(config, os.path.join(output_dir, "config.yaml"))
        if hasattr(train_dataset, "source_counts"):
            print("Dataset source counts: {}".format(train_dataset.source_counts), flush=True)
            if hasattr(train_dataset, "resolved_roots"):
                print("Dataset resolved roots: {}".format(train_dataset.resolved_roots), flush=True)

        if dataset_name == "sysu_ir_vcm_ir":
            batch_sampler = SYSUIRVCMIRSampler(
                dataset=train_dataset,
                sysu_ir_num_ids=config["train"].get("sysu_ir_num_ids", 8),
                vcm_ir_num_ids=config["train"].get("vcm_ir_num_ids", 4),
                num_instances=config["train"]["num_instances"],
                num_batches=config["train"]["steps_per_epoch"],
                seed=config["seed"],
            )
        elif dataset_name == "sysumm01" and config["dataset"].get("train_modality", "both") == "both":
            batch_sampler = CrossModalBatchSampler(
                dataset=train_dataset,
                num_ids=config["train"]["num_ids"],
                num_instances=config["train"]["num_instances"],
                num_batches=config["train"]["steps_per_epoch"],
                seed=config["seed"],
                rgb_instances=config["train"].get("rgb_instances"),
                ir_instances=config["train"].get("ir_instances"),
            )
        else:
            if config["train"].get("full_coverage_priority", True):
                batch_sampler = FullCoverageIdentityBatchSampler(
                    dataset=train_dataset,
                    num_ids=config["train"]["num_ids"],
                    num_instances=config["train"]["num_instances"],
                    num_batches=config["train"].get("steps_per_epoch"),
                    seed=config["seed"],
                    min_coverage=config["train"].get("min_epoch_coverage", 0.75),
                )
            else:
                batch_sampler = IdentityBatchSampler(
                    dataset=train_dataset,
                    num_ids=config["train"]["num_ids"],
                    num_instances=config["train"]["num_instances"],
                    num_batches=config["train"]["steps_per_epoch"],
                    seed=config["seed"],
                )

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=config["train"]["num_workers"],
            pin_memory=True,
            collate_fn=collate_sysu_ir_vcm_ir if dataset_name == "sysu_ir_vcm_ir" else None,
        )

        model = build_reid_model(model_config=config["model"], num_classes=train_dataset.num_classes)
        model.to(device)
        proto_memory = None
        memory_config = config["loss"].get("cross_modal_metric", {}).get("prototype_memory", {})
        if bool(memory_config.get("enabled", False)):
            proto_memory = CrossModalPrototypeMemory(
                num_classes=train_dataset.num_classes,
                feature_dim=getattr(model, "feature_dim", 768),
                momentum=memory_config.get("momentum", 0.9),
            ).to(device)
        init_checkpoint = config["train"].get("init_checkpoint")
        if init_checkpoint and not args.resume and config["model"].get("type") != "dual_encoder_align":
            init_msg = initialize_model_weights(model, init_checkpoint, init_config=config["train"])
            print(
                "Initialized model from {} (missing={}, unexpected={})".format(
                    init_checkpoint,
                    len(init_msg.missing_keys),
                    len(init_msg.unexpected_keys),
                ),
                flush=True,
            )

        ce_criterion = nn.CrossEntropyLoss()
        triplet_criterion = BatchHardTripletLoss(
            margin=config["loss"]["triplet_margin"],
            memory_size=config["loss"].get("cross_batch_memory_size", 0),
        )
        optimizer = AdamW(
            model.parameters(),
            lr=config["train"]["lr"],
            weight_decay=config["train"]["weight_decay"],
        )
        scheduler = build_scheduler(optimizer, config["train"])
        scaler = GradScaler(enabled=config["train"].get("amp", True) and device.type == "cuda")

        print(
            "LR scheduler: {} (base_lr={}, min_lr={}, warmup_epochs={}, warmup_init_lr={})".format(
                config["train"].get("scheduler", "cosine"),
                config["train"]["lr"],
                config["train"].get("min_lr", 1e-6),
                config["train"].get("warmup_epochs", 0),
                config["train"].get("warmup_init_lr", 0.0),
            ),
            flush=True,
        )

        start_epoch = 1
        best_map = -1.0
        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            if proto_memory is not None and "proto_memory" in checkpoint:
                proto_memory.load_state_dict(checkpoint["proto_memory"])
                proto_memory.to(device)
            start_epoch = checkpoint["epoch"] + 1
            best_map = checkpoint.get("best_map", -1.0)

        if args.eval_only:
            ckpt_path = args.checkpoint or args.resume
            if not ckpt_path:
                raise ValueError("--checkpoint or --resume is required with --eval-only")
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=True)
            for eval_mode in get_final_eval_modes(config):
                metrics, retrievals = evaluate_sysu(
                    model=model,
                    dataset_root=config["eval"].get("dataset_root", config["dataset"].get("root")),
                    image_size=tuple(config["dataset"]["image_size"]),
                    batch_size=config["eval"]["batch_size"],
                    num_workers=config["eval"]["num_workers"],
                    device=device,
                    mode=eval_mode,
                    num_trials=config["eval"]["num_trials"],
                        seed=config["seed"],
                        protocol=config["eval"].get("protocol", "cross_modality"),
                        modality=config["eval"].get("modality"),
                        **get_schp_eval_kwargs(config),
                    )
                dump_json(
                    {"metrics": metrics, "retrieval_examples": retrievals},
                    os.path.join(output_dir, "eval_{}_final.json".format(eval_mode)),
                )
            return

        fieldnames = [
            "epoch",
            "lr",
            "train_loss",
            "id_loss",
            "triplet_loss",
            "consistency_loss",
            "part_id_loss",
            "rgb_ir_clip_loss",
            "proto_align_loss",
            "cm_contrast_loss",
            "cm_triplet_loss",
            "memory_proto_loss",
            "cm_pos_dist",
            "cm_neg_dist",
            "cm_dist_gap",
            "all_mAP",
            "all_rank1",
            "epoch_seconds",
        ]
        print("Parameter count: {:.2f}M".format(count_parameters(model) / 1e6), flush=True)
        top_records = []
        save_top_k = int(config["train"].get("save_top_k", 0) or 0)

        for epoch in range(start_epoch, config["train"]["epochs"] + 1):
            model.train()
            freeze_backbone_epochs = int(config["train"].get("freeze_backbone_epochs", 0) or 0)
            freeze_backbone = epoch <= freeze_backbone_epochs
            last_blocks = config["train"].get("unfreeze_last_blocks")
            set_backbone_trainable(model, not freeze_backbone, last_blocks=last_blocks)
            if freeze_backbone:
                set_backbone_eval(model)
            if freeze_backbone_epochs > 0 and (epoch == 1 or epoch == freeze_backbone_epochs + 1):
                print(
                    "[Epoch {:03d}] backbone {}".format(
                        epoch,
                        "frozen" if freeze_backbone else "unfrozen",
                    ),
                    flush=True,
                )
            if hasattr(triplet_criterion, "reset_memory") and config["loss"].get("reset_memory_each_epoch", True):
                triplet_criterion.reset_memory()
            start_time = time.time()
            total_meter = AverageMeter()
            id_meter = AverageMeter()
            triplet_meter = AverageMeter()
            consistency_meter = AverageMeter()
            part_id_meter = AverageMeter()
            rgb_ir_clip_meter = AverageMeter()
            proto_align_meter = AverageMeter()
            cm_contrast_meter = AverageMeter()
            cm_triplet_meter = AverageMeter()
            memory_proto_meter = AverageMeter()
            cm_pos_dist_meter = AverageMeter()
            cm_neg_dist_meter = AverageMeter()
            cm_gap_meter = AverageMeter()
            num_steps = len(train_loader)

            print(
                "[Epoch {:03d}/{:03d}] start ({} batches)".format(epoch, config["train"]["epochs"], num_steps),
                flush=True,
            )

            for step, batch in enumerate(train_loader, start=1):
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                part_masks = batch.get("part_masks")
                if part_masks is not None:
                    part_masks = part_masks.to(device, non_blocking=True)
                tracklet_groups = batch.get("tracklet_group")
                if tracklet_groups is not None:
                    tracklet_groups = tracklet_groups.to(device, non_blocking=True)
                modalities = batch.get("modality")
                if modalities is not None:
                    modalities = modalities.to(device, non_blocking=True)
                optimizer.zero_grad()

                with autocast(enabled=scaler.is_enabled()):
                    outputs = model(images, part_masks=part_masks, modality=modalities)
                    id_loss, tri_loss = compute_reid_losses(
                        outputs,
                        labels,
                        batch,
                        ce_criterion,
                        triplet_criterion,
                        config["loss"],
                    )
                    consistency_weight = config["loss"].get("tracklet_consistency_weight", 0.0)
                    if consistency_weight > 0 and tracklet_groups is not None:
                        consistency_loss = tracklet_consistency_loss(outputs["embeddings"], tracklet_groups)
                    else:
                        consistency_loss = outputs["global_feat"].new_tensor(0.0)
                    part_id_loss = compute_part_id_loss(outputs, labels, ce_criterion, config["loss"])
                    rgb_ir_clip_loss, proto_align_loss = compute_rgb_ir_alignment_losses(
                        outputs,
                        labels,
                        batch,
                        config["loss"],
                        epoch,
                    )
                    (
                        cm_contrast_loss,
                        cm_triplet_loss,
                        memory_proto_loss,
                        cm_pos_dist,
                        cm_neg_dist,
                        cm_dist_gap,
                    ) = compute_strong_cross_modal_losses(
                        outputs,
                        labels,
                        batch,
                        config["loss"],
                        epoch,
                        proto_memory=proto_memory,
                    )
                    total_loss = (
                        id_loss
                        + tri_loss
                        + consistency_weight * consistency_loss
                        + part_id_loss
                        + rgb_ir_clip_loss
                        + proto_align_loss
                        + cm_contrast_loss
                        + cm_triplet_loss
                        + memory_proto_loss
                    )

                scaler.scale(total_loss).backward()
                if config["train"].get("clip_grad_norm", 0.0) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["clip_grad_norm"])
                scaler.step(optimizer)
                scaler.update()

                batch_size = images.size(0)
                total_meter.update(total_loss.item(), batch_size)
                id_meter.update(id_loss.item(), batch_size)
                triplet_meter.update(tri_loss.item(), batch_size)
                consistency_meter.update(consistency_loss.item(), batch_size)
                part_id_meter.update(part_id_loss.item(), batch_size)
                rgb_ir_clip_meter.update(rgb_ir_clip_loss.item(), batch_size)
                proto_align_meter.update(proto_align_loss.item(), batch_size)
                cm_contrast_meter.update(cm_contrast_loss.item(), batch_size)
                cm_triplet_meter.update(cm_triplet_loss.item(), batch_size)
                memory_proto_meter.update(memory_proto_loss.item(), batch_size)
                cm_pos_dist_meter.update(cm_pos_dist.item(), batch_size)
                cm_neg_dist_meter.update(cm_neg_dist.item(), batch_size)
                cm_gap_meter.update(cm_dist_gap.item(), batch_size)

                if step == 1 or step % max(args.print_freq, 1) == 0 or step == num_steps:
                    print(
                        "[Epoch {:03d}/{:03d}] step {:04d}/{:04d} lr={:.6g} loss={:.4f} id={:.4f} triplet={:.4f} cons={:.4f} part_id={:.4f} rgb_ir_clip={:.4f} proto={:.4f} cm_contrast={:.4f} cm_triplet={:.4f} mem_proto={:.4f} cm_gap={:.4f}".format(
                            epoch,
                            config["train"]["epochs"],
                            step,
                            num_steps,
                            optimizer.param_groups[0]["lr"],
                            total_meter.avg,
                            id_meter.avg,
                            triplet_meter.avg,
                            consistency_meter.avg,
                            part_id_meter.avg,
                            rgb_ir_clip_meter.avg,
                            proto_align_meter.avg,
                            cm_contrast_meter.avg,
                            cm_triplet_meter.avg,
                            memory_proto_meter.avg,
                            cm_gap_meter.avg,
                        ),
                        flush=True,
                    )

            scheduler.step()

            eval_trials = config["train"].get("eval_trials", 2)
            metrics = evaluate_and_save(model, config, output_dir, device, epoch, num_trials=eval_trials)
            epoch_seconds = time.time() - start_time
            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": total_meter.avg,
                "id_loss": id_meter.avg,
                "triplet_loss": triplet_meter.avg,
                "consistency_loss": consistency_meter.avg,
                "part_id_loss": part_id_meter.avg,
                "rgb_ir_clip_loss": rgb_ir_clip_meter.avg,
                "proto_align_loss": proto_align_meter.avg,
                "cm_contrast_loss": cm_contrast_meter.avg,
                "cm_triplet_loss": cm_triplet_meter.avg,
                "memory_proto_loss": memory_proto_meter.avg,
                "cm_pos_dist": cm_pos_dist_meter.avg,
                "cm_neg_dist": cm_neg_dist_meter.avg,
                "cm_dist_gap": cm_gap_meter.avg,
                "all_mAP": metrics["mAP"],
                "all_rank1": metrics["rank1"],
                "epoch_seconds": epoch_seconds,
            }
            append_metrics_row(os.path.join(output_dir, "history.csv"), fieldnames, row)
            print("[Epoch {:03d}/{:03d}] summary {}".format(epoch, config["train"]["epochs"], row), flush=True)

            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_map": best_map,
                "config": config,
            }
            if proto_memory is not None:
                state["proto_memory"] = proto_memory.state_dict()
            save_checkpoint(state, os.path.join(output_dir, "checkpoints", "last.pth"))
            top_records = update_topk_checkpoints(top_records, state, metrics, output_dir, save_top_k)

            if metrics["mAP"] > best_map:
                best_map = metrics["mAP"]
                state["best_map"] = best_map
                save_checkpoint(state, os.path.join(output_dir, "checkpoints", "best.pth"))

        best_checkpoint = os.path.join(output_dir, "checkpoints", "best.pth")
        checkpoint = torch.load(best_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        for eval_mode in get_final_eval_modes(config):
            metrics, retrievals = evaluate_sysu(
                model=model,
                dataset_root=config["eval"].get("dataset_root", config["dataset"].get("root")),
                image_size=tuple(config["dataset"]["image_size"]),
                batch_size=config["eval"]["batch_size"],
                num_workers=config["eval"]["num_workers"],
                device=device,
                mode=eval_mode,
                num_trials=config["eval"]["num_trials"],
                seed=config["seed"],
                protocol=config["eval"].get("protocol", "cross_modality"),
                modality=config["eval"].get("modality"),
                **get_schp_eval_kwargs(config),
            )
            dump_json(
                {"metrics": metrics, "retrieval_examples": retrievals},
                os.path.join(output_dir, "eval_{}_final.json".format(eval_mode)),
            )
        if config["train"].get("final_eval_top_k", False):
            evaluate_topk_checkpoints(model, config, output_dir, device, top_records)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


if __name__ == "__main__":
    main()
