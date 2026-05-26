#!/usr/bin/env python3
import argparse
from collections import OrderedDict
import torch


def unwrap(ckpt):
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "model_state", "teacher", "student"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def convert(sd):
    out = OrderedDict()
    for k, v in sd.items():
        nk = k
        if nk.startswith("base."):
            # TransReID checkpoints usually store ViT backbone under base.*
            nk = nk[len("base."):]
        elif nk.startswith("backbone.vit."):
            # Local project/sysumm01 ReIDModel checkpoints store the timm ViT
            # backbone under backbone.vit.*.
            nk = nk[len("backbone.vit."):]
        else:
            continue
        # Drop classifier / bnneck style heads if present
        if nk.startswith("classifier") or nk.startswith("bottleneck"):
            continue
        out[nk] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--state-key", default=None,
                    help="Optional key containing the model state dict, e.g. 'model'.")
    args = ap.parse_args()

    ckpt = torch.load(args.src, map_location="cpu")
    if args.state_key:
        if not isinstance(ckpt, dict) or args.state_key not in ckpt:
            raise KeyError(f"Checkpoint does not contain state key: {args.state_key}")
        sd = ckpt[args.state_key]
    else:
        sd = unwrap(ckpt)
    if not isinstance(sd, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(sd)}")

    conv = convert(sd)
    if not conv:
        raise RuntimeError("No convertible keys extracted. Check source checkpoint format.")

    torch.save(conv, args.dst)
    print(f"saved: {args.dst}")
    print(f"converted_keys: {len(conv)}")
    for i, k in enumerate(conv.keys()):
        if i >= 12:
            break
        print(k)


if __name__ == "__main__":
    main()
