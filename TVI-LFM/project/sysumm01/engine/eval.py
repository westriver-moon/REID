import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.sysumm01.engine.evaluator import evaluate_sysu
from project.sysumm01.models.reid_model import ReIDModel
from project.sysumm01.utils.config import dump_json, load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SYSU-MM01 experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", default="all", choices=["all", "indoor"])
    parser.add_argument("--num-trials", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    config["model"]["image_size"] = list(config["dataset"]["image_size"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = ReIDModel(config["model"], num_classes=config["dataset"]["num_classes"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    metrics, retrieval_examples = evaluate_sysu(
        model=model,
        dataset_root=config["dataset"]["root"],
        image_size=tuple(config["dataset"]["image_size"]),
        batch_size=config["eval"]["batch_size"],
        num_workers=config["eval"]["num_workers"],
        device=device,
        mode=args.mode,
        num_trials=args.num_trials or config["eval"]["num_trials"],
        seed=config["seed"],
    )
    payload = {"metrics": metrics, "retrieval_examples": retrieval_examples}
    print(payload)
    if args.output_json:
        dump_json(payload, args.output_json)


if __name__ == "__main__":
    main()
