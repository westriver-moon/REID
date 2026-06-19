# A1 PMT Recipe Implementation

Generated: 2026-06-19 CST.

## Config

Use:

```bash
python main.py --config_select config/stage_a/pmt_vit_stage_a_pmt_recipe.yaml
```

## What Changed From A1

- Keeps the Stage A image-only constraint: `training_mode: RGB_IR`, `joint_mode: image_only`.
- Keeps the TVI-LFM integration boundary: PMT_VIT still returns projected visual features for the existing classifier/evaluator.
- Adds a PMT-recipe training branch gated by `pmt_recipe: true`.
- Uses PMT-style transforms with `256 x 128` input.
- Uses a progressive visible branch:
  - epochs `0..5`: gray visible + IR
  - epochs `6..23`: RGB visible + IR
- Uses PMT losses:
  - gray stage: ID + PMT triplet
  - RGB stage: ID + PMT triplet + `0.5 * MSEL + 0.5 * DCL`
- Uses AdamW with PMT-style parameter grouping:
  - visual backbone patch/block LR factor: `0.5`
  - classifier/random adapter modules: base visual LR
- Uses cosine LR with warmup and per-group min LR factor:
  - `warmup_epochs: 3`
  - `target_lr_factor: 0.01`
  - `total_train_epoch: 24`

## Verification

- `python -m py_compile` passed for modified modules.
- PMT visual preflight passed with ImageNet checkpoint:
  - `Missing keys: 0`
  - `Unexpected keys: 0`
  - pos_embed resized from `[1,197,768]` to `[1,211,768]`
- Real SYSU loader batch passed:
  - `img_rgb_ori`, `img_rgb_aug`, `img_ir`: `(32, 3, 256, 128)`
  - no text fields loaded
  - visible/IR labels aligned by `num_pos`
- Full PMT_VIT real-batch backward smoke passed on GPU0.

## Notes

This is not a direct copy of the independent PMT training executable. It is a TVI-LFM Stage A integration that transfers the PMT training recipe while preserving the Stage A comparison surface and TVI-LFM evaluator.
