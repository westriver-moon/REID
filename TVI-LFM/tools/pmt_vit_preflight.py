from __future__ import annotations

import argparse
import os
import sys

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from network.pmt_vit_adapter import PMTViTVisual
from tools.utils import load_train_configs


def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight PMT ViT visual adapter preflight.")
    parser.add_argument("--config", default="config/stage_a/pmt_vit_stage_a.yaml")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--load-checkpoint", action="store_true")
    parser.add_argument("--skip-forward", action="store_true")
    parser.add_argument("--tiny-depth", action="store_true", help="Use depth=1 for a fast shape-only check.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_train_configs(args.config)
    depth = 1 if args.tiny_depth else config.pmt_depth
    pretrained_path = config.pmt_pretrained if args.load_checkpoint else None

    model = PMTViTVisual(
        input_resolution=config.img_size,
        patch_size=config.pmt_patch_size,
        stride_size=config.pmt_stride_size,
        embed_dim=config.pmt_embed_dim,
        depth=depth,
        num_heads=config.pmt_num_heads,
        mlp_ratio=config.pmt_mlp_ratio,
        drop_rate=config.pmt_dropout,
        attn_drop_rate=config.pmt_attention_dropout,
        drop_path_rate=config.pmt_drop_path_rate,
        output_dim=config.prj_output_dim,
        pretrained_path=pretrained_path,
        patch_embed_config=getattr(config, "pmt_patch_embed", None),
    )
    model.eval()

    print(f"pretrain_choice={config.pretrain_choice}")
    print(f"pmt_pretrained={config.pmt_pretrained}")
    print(f"load_checkpoint={args.load_checkpoint}")
    print(f"depth={depth}")
    print(f"num_patches={model.vit.patch_embed.num_patches}")
    if hasattr(model.vit.patch_embed, "branch_configs"):
        print(f"patch_branches={model.vit.patch_embed.branch_configs}")
    print(f"output_dim={model.output_dim}")

    if args.skip_forward:
        return

    dummy = torch.zeros(args.batch_size, 3, config.img_size[0], config.img_size[1])
    with torch.no_grad():
        output = model(dummy)
    print(f"tokens_shape={tuple(output['tokens'].shape)}")
    print(f"features_shape={tuple(output['features'].shape)}")


if __name__ == "__main__":
    main()
