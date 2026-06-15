from __future__ import annotations

import argparse
from pathlib import Path

from pmt_sysu.config import load_config, merge_overrides
from pmt_sysu.engine.trainer import train
from pmt_sysu.utils.logger import setup_logger
from pmt_sysu.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train independent PMT SYSU-MM01 baseline")
    parser.add_argument("--config", default="pmt_sysu/config/sysu_pmt.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke-batches", type=int, default=0)
    parser.add_argument("--override", action="append", default=[], help="Dotted key override, e.g. train.max_epoch=1")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    overrides = {}
    for item in args.override:
        key, value = item.split("=", 1)
        overrides[key] = value
    if args.no_amp:
        overrides["train.amp"] = False
    config = merge_overrides(config, overrides)

    data_root = Path(args.data_root or config.data.root).expanduser()
    pretrained = args.pretrained if args.pretrained is not None else config.model.pretrained
    output = Path(args.output or config.output.root).expanduser()
    logger = setup_logger(output)
    set_seed(int(config.seed))
    logger.info(f"data_root={data_root}")
    logger.info(f"output={output}")
    logger.info(f"device={args.device}")
    train(
        config=config,
        data_root=data_root,
        pretrained=pretrained,
        output_dir=output,
        device=args.device,
        resume=args.resume,
        weights=args.weights,
        smoke_batches=args.smoke_batches,
        logger=logger.info,
    )


if __name__ == "__main__":
    main()

