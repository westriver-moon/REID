from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pmt_sysu.config import load_config
from pmt_sysu.engine.evaluator import evaluate_sysu
from pmt_sysu.model import build_pmt_model
from pmt_sysu.utils.checkpoint import load_model_weights
from pmt_sysu.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate independent PMT SYSU-MM01 baseline")
    parser.add_argument("--config", default="pmt_sysu/config/sysu_pmt.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--mode", default="all", choices=["all", "indoor"])
    parser.add_argument("--gallery-mode", default="single", choices=["single", "multi"])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    data_root = Path(args.data_root or config.data.root).expanduser()
    output = Path(args.output or config.output.root).expanduser() / "evaluation_final"
    logger = setup_logger(output)
    model = build_pmt_model(config, num_classes=int(config.model.num_classes)).to(args.device)
    result = load_model_weights(model, args.weights, strict=False, map_location=args.device)
    logger.info(f"Loaded weights from {args.weights}; missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")
    model.eval()
    with torch.no_grad():
        evaluate_sysu(
            model,
            data_root,
            int(config.data.height),
            int(config.data.width),
            mode=args.mode,
            gallery_mode=args.gallery_mode,
            trials=args.trials,
            batch_size=int(config.test.batch_size),
            num_workers=int(config.test.num_workers),
            device=args.device,
            output_dir=output,
            logger=logger.info,
        )


if __name__ == "__main__":
    main()

