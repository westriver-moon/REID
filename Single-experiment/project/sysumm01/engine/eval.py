import argparse
import copy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.sysumm01.engine.evaluator import evaluate_sysu
from project.sysumm01.models.reid_model import build_reid_model
from project.sysumm01.utils.config import dump_json, load_config
from project.sysumm01.utils.misc import strip_prefix_if_present


def _extract_model_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def _infer_num_classes(config, state_dict):
    classifier_keys = (
        "classifier.weight",
        "module.classifier.weight",
    )
    for key in classifier_keys:
        if key in state_dict:
            return int(state_dict[key].shape[0])
    for key, value in state_dict.items():
        if (
            key.endswith("classifier.weight")
            and "part_classifier" not in key
            and "modality_classifier" not in key
            and hasattr(value, "shape")
            and len(value.shape) == 2
        ):
            return int(value.shape[0])
    if "dataset" in config and "num_classes" in config["dataset"]:
        return int(config["dataset"]["num_classes"])
    raise KeyError("Cannot infer num_classes: missing classifier.weight in checkpoint and dataset.num_classes in config")


def _get_dataset_root(config):
    dataset_root = config.get("eval", {}).get("dataset_root") or config.get("dataset", {}).get("root")
    if not dataset_root:
        raise KeyError("Evaluation requires eval.dataset_root or dataset.root")
    return dataset_root


def _get_schp_eval_kwargs(config):
    eval_config = config.get("eval", {})
    return {
        "schp_mask_root": eval_config.get("schp_mask_root"),
        "schp_min_part_pixels": eval_config.get("schp_min_part_pixels", 4),
        "schp_allow_fallback": eval_config.get("schp_allow_fallback", True),
        "schp_quality_index": eval_config.get("schp_quality_index"),
    }


def _get_final_seed(config):
    return int(config.get("eval", {}).get("final_seed", 0))


def _strip_eval_initializers(model_config):
    model_config = copy.deepcopy(model_config)

    def visit(node):
        if isinstance(node, dict):
            for key, value in list(node.items()):
                if key in ("pretrained_path", "init_checkpoint", "rgb_init_checkpoint", "ir_init_checkpoint"):
                    node[key] = None
                else:
                    visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(model_config)
    return model_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SYSU-MM01 experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", default="all", choices=["all", "indoor"])
    parser.add_argument("--num-trials", type=int, default=None)
    parser.add_argument("--id-split", default="test", choices=["train", "val", "test", "trainval"])
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    config["model"]["image_size"] = list(config["dataset"]["image_size"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = strip_prefix_if_present(_extract_model_state_dict(checkpoint), "module.")
    model = build_reid_model(_strip_eval_initializers(config["model"]), num_classes=_infer_num_classes(config, state_dict))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    metrics, retrieval_examples = evaluate_sysu(
        model=model,
        dataset_root=_get_dataset_root(config),
        image_size=tuple(config["dataset"]["image_size"]),
        batch_size=config["eval"]["batch_size"],
        num_workers=config["eval"]["num_workers"],
        device=device,
        mode=args.mode,
        num_trials=args.num_trials or config["eval"]["num_trials"],
        seed=_get_final_seed(config),
        protocol=config["eval"].get("protocol", "cross_modality"),
        modality=config["eval"].get("modality"),
        id_split=args.id_split,
        **_get_schp_eval_kwargs(config),
    )
    payload = {"metrics": metrics, "retrieval_examples": retrieval_examples}
    print(payload)
    if args.output_json:
        dump_json(payload, args.output_json)


if __name__ == "__main__":
    main()
