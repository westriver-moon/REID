from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda import amp
from torch.utils.data import DataLoader

from pmt_sysu.data.dataset import SYSUData
from pmt_sysu.data.sampler import PMTIdentitySampler, assert_pmt_batch_layout, build_label_positions
from pmt_sysu.data.transforms import build_transforms
from pmt_sysu.engine.evaluator import evaluate_sysu
from pmt_sysu.losses import DCL, MSEL, TripletLoss
from pmt_sysu.model import build_pmt_model
from pmt_sysu.config.defaults import to_plain_dict
from pmt_sysu.utils.checkpoint import load_model_weights, save_checkpoint
from pmt_sysu.utils.logger import AverageMeter
from pmt_sysu.utils.seed import get_random_state, set_random_state


def make_optimizer(config, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = float(config.train.base_lr)
        weight_decay = float(config.train.weight_decay)
        if "bias" in key:
            lr *= float(config.train.bias_lr_factor)
            weight_decay = float(config.train.bias_weight_decay)
        if key.startswith("base.patch_embed.") or "base.blocks." in key:
            lr *= float(config.train.backbone_lr_factor)
        params.append({"params": [value], "lr": lr, "initial_lr": lr, "weight_decay": weight_decay, "name": key})
    return torch.optim.AdamW(params, lr=float(config.train.base_lr), weight_decay=float(config.train.weight_decay))


def set_cosine_lr(config, optimizer, epoch: int):
    max_epoch = int(config.train.max_epoch)
    warmup_epochs = int(config.train.warmup_epochs)
    warmup_factor = float(config.train.warmup_factor)
    min_lr_factor = float(config.train.min_lr_factor)
    if epoch <= warmup_epochs:
        factor = warmup_factor + (1 - warmup_factor) * epoch / max(warmup_epochs, 1)
    else:
        progress = (epoch - warmup_epochs) / max(max_epoch - warmup_epochs, 1)
        factor = min_lr_factor + 0.5 * (1 - min_lr_factor) * (1 + torch.cos(torch.tensor(progress * torch.pi))).item()
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * factor
    return factor


def build_train_datasets(config, data_root):
    transforms = build_transforms(int(config.data.height), int(config.data.width))
    gray = SYSUData(data_root, transform_visible=transforms["rgb2gray"], transform_ir=transforms["thermal"])
    rgb = SYSUData(data_root, transform_visible=transforms["rgb"], transform_ir=transforms["thermal"])
    color_pos = build_label_positions(rgb.train_color_label)
    thermal_pos = build_label_positions(rgb.train_thermal_label)
    return gray, rgb, color_pos, thermal_pos


def build_epoch_loader(config, dataset, color_pos, thermal_pos):
    sampler = PMTIdentitySampler(
        dataset.train_color_label,
        dataset.train_thermal_label,
        color_pos,
        thermal_pos,
        batch_size=int(config.data.batch_size_per_modality),
        num_pos=int(config.data.num_pos),
    )
    dataset.set_indices(sampler.index1, sampler.index2)
    return DataLoader(
        dataset,
        batch_size=int(config.data.batch_size_per_modality),
        sampler=sampler,
        num_workers=int(config.data.num_workers),
        drop_last=True,
        pin_memory=True,
    )


def compute_pmt_losses(config, model, batch, device, epoch: int, criterion_id, criterion_tri, criterion_msel, criterion_dcl):
    input_visible, input_ir, label_visible, label_ir = batch
    input_visible = input_visible.to(device, non_blocking=True)
    input_ir = input_ir.to(device, non_blocking=True)
    label_visible = label_visible.to(device, non_blocking=True).long()
    label_ir = label_ir.to(device, non_blocking=True).long()
    batch_size = int(config.data.batch_size_per_modality)
    num_pos = int(config.data.num_pos)
    assert_pmt_batch_layout(label_visible, label_ir, num_pos=num_pos, batch_size=batch_size)

    outputs = model(torch.cat([input_visible, input_ir], dim=0), return_dict=True)
    logits = outputs["logits"]
    features = outputs["features"]
    assert features.shape == (batch_size * 2, int(config.model.embed_dim))
    labels = torch.cat([label_visible, label_ir], dim=0)
    score_visible, score_ir = logits.chunk(2, dim=0)
    feat_visible, feat_ir = features.chunk(2, dim=0)

    loss_id = criterion_id(score_visible, label_visible) + criterion_id(score_ir, label_ir)
    if epoch <= int(config.train.progressive_epoch):
        stage = "gray_ir"
        loss_tri = criterion_tri(feat_visible, feat_visible, label_visible) + criterion_tri(feat_ir, feat_ir, label_ir)
        loss_msel = features.new_zeros(())
        loss_dcl = features.new_zeros(())
        loss = loss_id + loss_tri
    else:
        stage = "rgb_ir"
        loss_tri = criterion_tri(features, features, labels)
        loss_msel = criterion_msel(features, labels)
        loss_dcl = criterion_dcl(features, labels)
        loss = loss_id + loss_tri + float(config.train.msel_weight) * loss_msel + float(config.train.dcl_weight) * loss_dcl

    assert torch.isfinite(loss), "total loss is not finite"
    assert torch.isfinite(loss_id), "id loss is not finite"
    assert torch.isfinite(loss_tri), "triplet loss is not finite"
    assert torch.isfinite(loss_msel), "MSEL loss is not finite"
    assert torch.isfinite(loss_dcl), "DCL loss is not finite"
    acc_visible = (score_visible.argmax(dim=1) == label_visible).float().mean()
    acc_ir = (score_ir.argmax(dim=1) == label_ir).float().mean()
    return {
        "loss": loss,
        "stage": stage,
        "id_loss": loss_id,
        "triplet_loss": loss_tri,
        "msel_loss": loss_msel,
        "dcl_loss": loss_dcl,
        "visible_acc": acc_visible,
        "ir_acc": acc_ir,
        "features": features,
    }


def _append_metrics(output_dir: Path, row: dict):
    json_path = output_dir / "metrics.jsonl"
    with json_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    csv_path = output_dir / "metrics.csv"
    exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train(config, data_root, pretrained, output_dir, device, resume=None, weights=None, smoke_batches=0, logger=print):
    output_dir = Path(output_dir)
    ckpt_dir = output_dir / "checkpoints"
    eval_dir = output_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "config_resolved.yaml").open("w", encoding="utf-8") as handle:
        import yaml

        yaml.safe_dump(to_plain_dict(config), handle, allow_unicode=True, sort_keys=False)

    gray_dataset, rgb_dataset, color_pos, thermal_pos = build_train_datasets(config, data_root)
    num_classes = len(set(rgb_dataset.train_color_label.tolist()))
    model = build_pmt_model(config, num_classes=num_classes).to(device)
    if pretrained:
        model.load_imagenet_pretrained(pretrained, logger=logger)
    if weights:
        result = load_model_weights(model, weights, strict=False)
        logger(f"Loaded model weights from {weights}; missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")

    criterion_id = nn.CrossEntropyLoss()
    criterion_tri = TripletLoss(margin=float(config.train.triplet_margin), feat_norm="no")
    criterion_msel = MSEL(num_pos=int(config.data.num_pos), feat_norm="no")
    criterion_dcl = DCL(num_pos=int(config.data.num_pos), feat_norm="no")
    optimizer = make_optimizer(config, model)
    scaler = amp.GradScaler(enabled=bool(config.train.amp) and torch.cuda.is_available())
    best_mAP = 0.0
    start_epoch = int(config.train.start_epoch)

    if resume:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        best_mAP = float(checkpoint.get("best_mAP", 0.0))
        start_epoch = int(checkpoint["epoch"]) + 1
        set_random_state(checkpoint.get("random_state", {}))
        logger(f"Resumed from {resume} at epoch {start_epoch}")

    max_epoch = int(config.train.max_epoch)
    if smoke_batches:
        max_epoch = min(max_epoch, start_epoch)

    for epoch in range(start_epoch, max_epoch + 1):
        lr_factor = set_cosine_lr(config, optimizer, epoch)
        is_gray_stage = epoch <= int(config.train.progressive_epoch)
        dataset = gray_dataset if is_gray_stage else rgb_dataset
        train_loader = build_epoch_loader(config, dataset, color_pos, thermal_pos)
        model.train()
        meters = {name: AverageMeter() for name in ["loss", "id", "tri", "msel", "dcl", "acc_v", "acc_ir"]}
        start = time.time()
        for iteration, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(enabled=bool(config.train.amp) and torch.cuda.is_available()):
                out = compute_pmt_losses(
                    config,
                    model,
                    batch,
                    device,
                    epoch,
                    criterion_id,
                    criterion_tri,
                    criterion_msel,
                    criterion_dcl,
                )
            scaler.scale(out["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()

            meters["loss"].update(out["loss"].item())
            meters["id"].update(out["id_loss"].item())
            meters["tri"].update(out["triplet_loss"].item())
            meters["msel"].update(out["msel_loss"].item())
            meters["dcl"].update(out["dcl_loss"].item())
            meters["acc_v"].update(out["visible_acc"].item())
            meters["acc_ir"].update(out["ir_acc"].item())
            if iteration % int(config.train.log_interval) == 0 or smoke_batches:
                logger(
                    f"epoch={epoch} iter={iteration}/{len(train_loader)} stage={out['stage']} "
                    f"loss={meters['loss'].avg:.4f} id={meters['id'].avg:.4f} tri={meters['tri'].avg:.4f} "
                    f"msel={meters['msel'].avg:.4f} dcl={meters['dcl'].avg:.4f} "
                    f"acc_visible={meters['acc_v'].avg:.4f} acc_ir={meters['acc_ir'].avg:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e}"
                )
            if smoke_batches and iteration >= smoke_batches:
                break

        elapsed = time.time() - start
        row = {
            "epoch": epoch,
            "stage": "gray_ir" if is_gray_stage else "rgb_ir",
            "lr": optimizer.param_groups[0]["lr"],
            "lr_factor": lr_factor,
            "total_loss": meters["loss"].avg,
            "id_loss": meters["id"].avg,
            "triplet_loss": meters["tri"].avg,
            "msel_loss": meters["msel"].avg,
            "dcl_loss": meters["dcl"].avg,
            "visible_acc": meters["acc_v"].avg,
            "ir_acc": meters["acc_ir"].avg,
            "rank1": "",
            "mAP": "",
            "mINP": "",
            "epoch_time_sec": elapsed,
        }

        if not smoke_batches and epoch % int(config.train.eval_interval) == 0:
            average, _ = evaluate_sysu(
                model,
                data_root,
                int(config.data.height),
                int(config.data.width),
                mode=config.test.mode,
                gallery_mode=config.test.gallery_mode,
                trials=1,
                batch_size=int(config.test.batch_size),
                num_workers=int(config.test.num_workers),
                device=device,
                output_dir=eval_dir / f"epoch_{epoch:02d}",
                logger=logger,
            )
            row.update({"rank1": average["rank1"], "mAP": average["mAP"], "mINP": average["mINP"]})
            if average["mAP"] > best_mAP:
                best_mAP = average["mAP"]
                shutil.copyfile(ckpt_dir / "latest.pth", ckpt_dir / "best.pth") if (ckpt_dir / "latest.pth").exists() else None

        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": {"kind": "manual_cosine", "epoch": epoch},
            "scaler_state_dict": scaler.state_dict(),
            "best_mAP": best_mAP,
            "config": to_plain_dict(config),
            "random_state": get_random_state(),
        }
        save_checkpoint(ckpt_dir / "latest.pth", payload)
        if not smoke_batches and epoch % int(config.train.save_interval) == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch:02d}.pth", payload)
        if row["mAP"] != "" and row["mAP"] == best_mAP:
            save_checkpoint(ckpt_dir / "best.pth", payload)
        _append_metrics(output_dir, row)
        if smoke_batches:
            logger("Smoke training completed without writing a formal epoch checkpoint beyond latest.pth")
            break

    return model
