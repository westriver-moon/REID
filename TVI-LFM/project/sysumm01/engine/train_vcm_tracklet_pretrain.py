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

from project.sysumm01.datasets.vcm import VCMTrackletDataset, IdentityModalityBalancedTrackletSampler
from project.sysumm01.engine.train import (
    TeeStream,
    WarmupCosineScheduler,
    cross_modal_batch_hard_triplet_loss,
    cross_modal_contrast_loss,
    initialize_model_weights,
    set_backbone_eval,
    set_backbone_trainable,
)
from project.sysumm01.models.reid_model import build_reid_model
from project.sysumm01.utils.config import dump_config, dump_json, load_config
from project.sysumm01.utils.misc import AverageMeter, append_metrics_row, count_parameters, ensure_dir, save_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train HITSZ-VCM RGB-IR tracklet pretraining")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--print-freq", type=int, default=20)
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


def build_scheduler(optimizer, train_config):
    scheduler_name = train_config.get("scheduler", "warmup_cosine")
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
            min_lr=train_config.get("min_lr", 1e-6),
            warmup_epochs=train_config.get("warmup_epochs", 4),
            warmup_init_lr=train_config.get("warmup_init_lr", 1e-6),
        )
    raise ValueError("Unsupported scheduler: {}".format(scheduler_name))


def collate_vcm_tracklets(batch):
    images = torch.stack([item["images"] for item in batch], dim=0)
    labels = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
    pids = torch.tensor([int(item["pid"]) for item in batch], dtype=torch.long)
    camids = torch.tensor([int(item["camid"]) for item in batch], dtype=torch.long)
    modalities = torch.tensor([int(item["modality"]) for item in batch], dtype=torch.long)
    return {
        "images": images,
        "label": labels,
        "pid": pids,
        "camid": camids,
        "modality": modalities,
        "tracklet_id": [item["tracklet_id"] for item in batch],
        "frame_paths": [item["frame_paths"] for item in batch],
    }


def tracklet_consistency_loss(frame_embeddings, tracklet_embeddings):
    if frame_embeddings.shape[1] <= 1:
        return frame_embeddings.new_tensor(0.0)
    center = F.normalize(tracklet_embeddings, dim=1).unsqueeze(1)
    frame_embeddings = F.normalize(frame_embeddings, dim=2)
    return 1.0 - (frame_embeddings * center).sum(dim=2).mean()


def aggregate_outputs(outputs, batch_size, frames_per_tracklet):
    frame_logits = outputs["logits"].reshape(batch_size, frames_per_tracklet, -1)
    frame_embeddings = outputs["embeddings"].reshape(batch_size, frames_per_tracklet, -1)
    tracklet_logits = frame_logits.mean(dim=1)
    tracklet_embeddings = F.normalize(frame_embeddings.mean(dim=1), dim=1)
    return tracklet_logits, tracklet_embeddings, frame_embeddings


def main():
    args = parse_args()
    config = load_config(args.config, overrides=parse_config_overrides(args.overrides))
    if args.seed is not None:
        config["seed"] = args.seed
    config["model"]["image_size"] = list(config["dataset"]["image_size"])

    output_dir = args.output or config.get("train", {}).get("output_dir")
    if not output_dir:
        raise ValueError("Output directory must be provided by --output or train.output_dir")
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, "checkpoints"))
    dump_config(config, os.path.join(output_dir, "config.yaml"))

    log_path = os.path.join(output_dir, "train.log")
    log_handle, original_stdout, original_stderr = install_stream_tee(log_path)
    try:
        print("Logging to {}".format(log_path), flush=True)
        set_seed(config["seed"])
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        dataset = VCMTrackletDataset(
            root=config["dataset"]["root"],
            tracklet_json=config["dataset"]["index_path"],
            image_size=tuple(config["dataset"]["image_size"]),
            frames_per_tracklet=config["dataset"].get("frames_per_tracklet", 2),
            mode=config["dataset"].get("mode", "rgb_ir"),
            frame_sampling=config["dataset"].get("frame_sampling", "random"),
            train_augment=config["dataset"].get("train_augment", "strong_reid"),
        )
        config["dataset"]["num_classes"] = dataset.num_classes
        dump_config(config, os.path.join(output_dir, "config.yaml"))
        print("VCM source counts: {}".format(dataset.source_counts), flush=True)
        print("VCM metadata: {}".format(dataset.metadata), flush=True)
        effective_images_per_batch = (
            config["train"]["num_ids"]
            * (config["train"]["rgb_tracklets_per_id"] + config["train"]["ir_tracklets_per_id"])
            * config["dataset"]["frames_per_tracklet"]
        )
        effective_images_per_epoch = len(dataset.tracklets) * config["dataset"]["frames_per_tracklet"]
        print(
            "VCM pretrain setup: K={}, identities/batch={}, rgb_tracklets/id={}, ir_tracklets/id={}, "
            "effective_images/batch={}, effective_images/epoch={}".format(
                config["dataset"]["frames_per_tracklet"],
                config["train"]["num_ids"],
                config["train"]["rgb_tracklets_per_id"],
                config["train"]["ir_tracklets_per_id"],
                effective_images_per_batch,
                effective_images_per_epoch,
            ),
            flush=True,
        )

        sampler = IdentityModalityBalancedTrackletSampler(
            dataset=dataset,
            num_ids=config["train"]["num_ids"],
            num_rgb_tracklets=config["train"]["rgb_tracklets_per_id"],
            num_ir_tracklets=config["train"]["ir_tracklets_per_id"],
            num_batches=config["train"]["steps_per_epoch"],
            seed=config["seed"],
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=config["train"]["num_workers"],
            pin_memory=True,
            collate_fn=collate_vcm_tracklets,
        )

        model = build_reid_model(config["model"], num_classes=dataset.num_classes)
        model.to(device)
        init_checkpoint = config["train"].get("init_checkpoint")
        if init_checkpoint and not args.resume:
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
        optimizer = AdamW(
            model.parameters(),
            lr=config["train"]["lr"],
            weight_decay=config["train"]["weight_decay"],
        )
        scheduler = build_scheduler(optimizer, config["train"])
        scaler = GradScaler(enabled=config["train"].get("amp", True) and device.type == "cuda")

        start_epoch = 1
        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = int(checkpoint["epoch"]) + 1

        fieldnames = [
            "epoch",
            "lr",
            "train_loss",
            "id_loss",
            "cm_contrast_loss",
            "cm_triplet_loss",
            "consistency_loss",
            "cm_pos_dist",
            "cm_neg_dist",
            "cm_gap",
            "epoch_seconds",
        ]
        print("Parameter count: {:.2f}M".format(count_parameters(model) / 1e6), flush=True)

        for epoch in range(start_epoch, config["train"]["epochs"] + 1):
            model.train()
            freeze_backbone_epochs = int(config["train"].get("freeze_backbone_epochs", 0) or 0)
            freeze_backbone = epoch <= freeze_backbone_epochs
            set_backbone_trainable(model, not freeze_backbone, last_blocks=config["train"].get("unfreeze_last_blocks"))
            if freeze_backbone:
                set_backbone_eval(model)
            total_meter = AverageMeter()
            id_meter = AverageMeter()
            cm_contrast_meter = AverageMeter()
            cm_triplet_meter = AverageMeter()
            consistency_meter = AverageMeter()
            cm_pos_meter = AverageMeter()
            cm_neg_meter = AverageMeter()
            cm_gap_meter = AverageMeter()
            start_time = time.time()
            print("[Epoch {:03d}/{:03d}] start ({} batches)".format(epoch, config["train"]["epochs"], len(loader)), flush=True)

            for step, batch in enumerate(loader, start=1):
                images = batch["images"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                modalities = batch["modality"].to(device, non_blocking=True)
                batch_size, frames_per_tracklet, channels, height, width = images.shape
                flat_images = images.reshape(batch_size * frames_per_tracklet, channels, height, width)
                flat_modalities = modalities.repeat_interleave(frames_per_tracklet)

                optimizer.zero_grad()
                with autocast(enabled=scaler.is_enabled()):
                    outputs = model(flat_images, modality=flat_modalities)
                    tracklet_logits, tracklet_embeddings, frame_embeddings = aggregate_outputs(
                        outputs,
                        batch_size=batch_size,
                        frames_per_tracklet=frames_per_tracklet,
                    )
                    id_loss = float(config["loss"].get("id_weight", 1.0)) * ce_criterion(tracklet_logits, labels)
                    contrast_weight = float(config["loss"].get("cm_contrast_weight", 0.0))
                    if contrast_weight > 0:
                        cm_contrast_loss = contrast_weight * cross_modal_contrast_loss(
                            tracklet_embeddings,
                            labels,
                            modalities,
                            temperature=config["loss"].get("temperature", 0.07),
                        )
                    else:
                        cm_contrast_loss = tracklet_embeddings.new_tensor(0.0)
                    cm_triplet_loss = float(config["loss"].get("cm_triplet_weight", 0.0)) * cross_modal_batch_hard_triplet_loss(
                        tracklet_embeddings,
                        labels,
                        modalities,
                        margin=config["loss"].get("triplet_margin", 0.3),
                    )
                    consistency_loss = float(config["loss"].get("tracklet_consistency_weight", 0.0)) * tracklet_consistency_loss(
                        frame_embeddings,
                        tracklet_embeddings,
                    )
                    with torch.no_grad():
                        rgb_mask = modalities == 0
                        ir_mask = modalities == 1
                        if rgb_mask.any() and ir_mask.any():
                            dist = torch.cdist(tracklet_embeddings[rgb_mask], tracklet_embeddings[ir_mask], p=2)
                            positive = labels[rgb_mask][:, None].eq(labels[ir_mask][None, :])
                            negative = ~positive
                            pos = dist[positive].mean() if positive.any() else dist.new_tensor(0.0)
                            neg = dist[negative].mean() if negative.any() else dist.new_tensor(0.0)
                            gap = neg - pos
                        else:
                            pos = neg = gap = tracklet_embeddings.new_tensor(0.0)
                    total_loss = id_loss + cm_contrast_loss + cm_triplet_loss + consistency_loss

                scaler.scale(total_loss).backward()
                if config["train"].get("clip_grad_norm", 0.0) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["clip_grad_norm"])
                scaler.step(optimizer)
                scaler.update()

                total_meter.update(total_loss.item(), batch_size)
                id_meter.update(id_loss.item(), batch_size)
                cm_contrast_meter.update(cm_contrast_loss.item(), batch_size)
                cm_triplet_meter.update(cm_triplet_loss.item(), batch_size)
                consistency_meter.update(consistency_loss.item(), batch_size)
                cm_pos_meter.update(pos.item(), batch_size)
                cm_neg_meter.update(neg.item(), batch_size)
                cm_gap_meter.update(gap.item(), batch_size)

                if step == 1 or step % max(args.print_freq, 1) == 0 or step == len(loader):
                    print(
                        "[Epoch {:03d}/{:03d}] step {:04d}/{:04d} lr={:.6g} loss={:.4f} id={:.4f} "
                        "cm_contrast={:.4f} cm_tri={:.4f} cons={:.4f} cm_gap={:.4f}".format(
                            epoch,
                            config["train"]["epochs"],
                            step,
                            len(loader),
                            optimizer.param_groups[0]["lr"],
                            total_meter.avg,
                            id_meter.avg,
                            cm_contrast_meter.avg,
                            cm_triplet_meter.avg,
                            consistency_meter.avg,
                            cm_gap_meter.avg,
                        ),
                        flush=True,
                    )

            scheduler.step()
            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": total_meter.avg,
                "id_loss": id_meter.avg,
                "cm_contrast_loss": cm_contrast_meter.avg,
                "cm_triplet_loss": cm_triplet_meter.avg,
                "consistency_loss": consistency_meter.avg,
                "cm_pos_dist": cm_pos_meter.avg,
                "cm_neg_dist": cm_neg_meter.avg,
                "cm_gap": cm_gap_meter.avg,
                "epoch_seconds": time.time() - start_time,
            }
            append_metrics_row(os.path.join(output_dir, "history.csv"), fieldnames, row)
            print("[Epoch {:03d}/{:03d}] summary {}".format(epoch, config["train"]["epochs"], row), flush=True)
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config,
                "num_classes": dataset.num_classes,
                "dataset": "HITSZ-VCM",
                "pretrain_type": "rgb_ir_tracklet",
                "frames_per_tracklet": config["dataset"]["frames_per_tracklet"],
            }
            save_checkpoint(state, os.path.join(output_dir, "checkpoints", "last.pth"))

        save_checkpoint(state, os.path.join(output_dir, "checkpoints", "vcm_rgb_ir_tracklet_last.pth"))
        dump_json({"last_epoch": config["train"]["epochs"], "source_counts": dataset.source_counts}, os.path.join(output_dir, "summary.json"))
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


if __name__ == "__main__":
    main()
