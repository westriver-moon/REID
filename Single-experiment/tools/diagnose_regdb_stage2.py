#!/usr/bin/env python3
import sys

import torch

from config.config_rn import get_args
from core import build_model
from data_loader.loader import Loader
from main import _extract_model_state_dict, seed_torch


def main():
    sys.argv = [
        "diagnose_regdb_stage2.py",
        "--mode",
        "train",
        "--dataset",
        "regdb",
        "--regdb_data_path",
        "/home/cgv841/datasets/RegDB/",
        "--trial",
        "1",
        "--pretrain_choice",
        "LASTVIT_ORI",
        "--lastvit_pretrained",
        "/home/cgv841/ybj/pretrained/ViT_190k.pth",
        "--lastvit_pretrained_rgb",
        "/home/cgv841/ybj/pretrained/external/rgb_sysumm01_vitb_best_timm.pth",
        "--lastvit_pretrained_ir",
        "/home/cgv841/ybj/pretrained/external/ir_sysumm01_vitb_best_timm.pth",
        "--training_mode",
        "RGB_IR",
        "--loss_names",
        "id,wrt,clip,proto",
        "--enable_rgb_ir_clip",
        "--enable_proto_align",
        "--clip_use_aug_rgb",
        "--batch-size",
        "8",
        "--num_workers",
        "0",
        "--gpu_id",
        "0",
        "--CUDA_VISIBLE_DEVICES",
        "0",
    ]
    config = get_args()
    seed_torch(config.seed)
    config.pid_num = 206
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loaders = Loader(config)
    model = build_model(config)
    ckpt_path = (
        "logs/regdb_pair_clip_warmup_b16/regdb/Base/"
        "Baseline_1_train[RGB_IR]_pair_clip/checkpoint/checkpoint_latest.pth"
    )
    checkpoint = torch.load(ckpt_path, map_location=device)
    state = _extract_model_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.train()

    print(f"device={device}")
    print(f"missing={len(missing)} unexpected={len(unexpected)}")
    print(
        "logit_scale_param={:.6f} logit_scale_exp={:.6g}".format(
            float(model.logit_scale.detach().cpu()),
            float(model.logit_scale.exp().detach().cpu()),
        )
    )

    batch = next(iter(loaders.get_train_loader()))
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        ret = model(batch, mode="1/3")

    total = None
    for key, value in ret.items():
        if not torch.is_tensor(value):
            print(f"{key}: {value}")
            continue
        finite = bool(torch.isfinite(value).all().detach().cpu())
        scalar = float(value.detach().float().cpu()) if value.numel() == 1 else None
        print(f"{key}: value={scalar} finite={finite}")
        if "loss" in key:
            total = value if total is None else total + value
    print(f"total_loss={float(total.detach().float().cpu())} finite={bool(torch.isfinite(total).detach().cpu())}")


if __name__ == "__main__":
    main()
