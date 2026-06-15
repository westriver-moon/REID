from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.cuda import amp

from pmt_sysu.config import load_config
from pmt_sysu.data.sampler import assert_pmt_batch_layout
from pmt_sysu.engine.trainer import (
    build_epoch_loader,
    build_train_datasets,
    compute_pmt_losses,
)
from pmt_sysu.losses import DCL, MSEL, TripletLoss
from pmt_sysu.model import build_pmt_model


def parse_args():
    parser = argparse.ArgumentParser(description="Preflight PMT SYSU pipeline")
    parser.add_argument("--config", default="pmt_sysu/config/sysu_pmt.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-missing-pretrained", action="store_true")
    return parser.parse_args()


def check_data_files(root: Path):
    required = [
        "train_rgb_resized_img.npy",
        "train_rgb_resized_label.npy",
        "train_ir_resized_img.npy",
        "train_ir_resized_label.npy",
        "exp/train_id.txt",
        "exp/val_id.txt",
        "exp/test_id.txt",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing SYSU files: {missing}")
    print("data files: OK")


def main():
    args = parse_args()
    config = load_config(args.config)
    data_root = Path(args.data_root or config.data.root)
    pretrained = Path(args.pretrained or config.model.pretrained)
    check_data_files(data_root)
    if not pretrained.is_file():
        if not args.allow_missing_pretrained:
            raise FileNotFoundError(f"Pretrained weight missing: {pretrained}")
        print(f"pretrained missing, continue with random init because --allow-missing-pretrained was set: {pretrained}")

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    gray_dataset, rgb_dataset, color_pos, thermal_pos = build_train_datasets(config, data_root)
    print(f"train RGB images={len(rgb_dataset.train_color_label)} IR images={len(rgb_dataset.train_thermal_label)}")

    model = build_pmt_model(config).to(device)
    if pretrained.is_file():
        model.load_imagenet_pretrained(pretrained)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params total={total_params:,} trainable={trainable_params:,}")

    criterion_id = torch.nn.CrossEntropyLoss()
    criterion_tri = TripletLoss(margin=float(config.train.triplet_margin), feat_norm="no")
    criterion_msel = MSEL(num_pos=int(config.data.num_pos), feat_norm="no")
    criterion_dcl = DCL(num_pos=int(config.data.num_pos), feat_norm="no")

    model.train()
    for epoch, dataset, expected_stage in [(1, gray_dataset, "gray_ir"), (int(config.train.progressive_epoch) + 1, rgb_dataset, "rgb_ir")]:
        loader = build_epoch_loader(config, dataset, color_pos, thermal_pos)
        batch = next(iter(loader))
        label_visible, label_ir = batch[2], batch[3]
        assert_pmt_batch_layout(label_visible, label_ir, int(config.data.num_pos), int(config.data.batch_size_per_modality))
        with amp.autocast(enabled=device.type == "cuda"):
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
        assert out["stage"] == expected_stage
        assert out["features"].shape == (int(config.data.batch_size_per_modality) * 2, int(config.model.embed_dim))
        out["loss"].backward()
        has_backbone_grad = any(
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            for n, p in model.named_parameters()
            if n.startswith("base.")
        )
        has_classifier_grad = any(
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            for n, p in model.named_parameters()
            if n.startswith("classifier.")
        )
        assert has_backbone_grad, "backbone has no effective gradient"
        assert has_classifier_grad, "classifier has no effective gradient"
        model.zero_grad(set_to_none=True)
        print(
            f"epoch={epoch} stage={out['stage']} features={tuple(out['features'].shape)} "
            f"loss={out['loss'].item():.4f} id={out['id_loss'].item():.4f} "
            f"tri={out['triplet_loss'].item():.4f} msel={out['msel_loss'].item():.4f} dcl={out['dcl_loss'].item():.4f}"
        )

    if device.type == "cuda":
        print(f"cuda memory allocated={torch.cuda.memory_allocated(device) / 1024 ** 2:.1f} MiB")
    print("PMT preflight passed.")


if __name__ == "__main__":
    main()
