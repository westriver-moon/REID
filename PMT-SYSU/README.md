# Independent PMT SYSU-MM01 Baseline

This directory is an independent reproduction baseline for:

Learning Progressive Modality-shared Transformers for Effective Visible-Infrared Person Re-identification, AAAI 2023.

Official references:

- Paper: https://arxiv.org/abs/2212.00226
- Official PMT repository: https://github.com/hulu88/PMT
- Official PyTorch code: https://github.com/hulu88/PMT/tree/main/Pytorch-PMT-VI-ReID
- SYSU-MM01 page: https://isee.sysu.edu.cn/project/RGBIRReID.htm

This implementation is intentionally separate from `TVI-LFM`. It does not use CLIP, text descriptions, LASTViT, VCM, RegDB pretraining, or TVI-LFM fusion code.

## Data

Default SYSU root:

```bash
/home/cgv841/datasets/SYSU-MM01
```

Required files:

```text
train_rgb_resized_img.npy
train_rgb_resized_label.npy
train_ir_resized_img.npy
train_ir_resized_label.npy
exp/train_id.txt
exp/val_id.txt
exp/test_id.txt
```

If the `.npy` files are absent, refer to the official PMT preprocessing script:

https://github.com/hulu88/PMT/blob/main/Pytorch-PMT-VI-ReID/process_sysu.py

Do not reprocess SYSU when the existing cache is already valid.

## Weights

ImageNet ViT-B/16:

```bash
python -m pmt_sysu.tools.download_weights --imagenet
```

Official PMT SYSU checkpoint:

```bash
python -m pip install gdown
python -m pmt_sysu.tools.download_weights --official
```

Manual URLs:

- ImageNet ViT-B/16: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth
- Official PMT SYSU checkpoint: https://drive.google.com/file/d/1S7Upn_8dWHNN5R3woazpocFU6J8hvCIe/view?usp=share_link

Weights and datasets must not be committed.

## Preflight

```bash
python -m pmt_sysu.tools.preflight \
  --config pmt_sysu/config/sysu_pmt.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --pretrained pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  --device cuda:0
```

For pipeline-only checks without ImageNet weights:

```bash
python -m pmt_sysu.tools.preflight \
  --config pmt_sysu/config/sysu_pmt.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --allow-missing-pretrained \
  --device cuda:0
```

## Train

```bash
python -m pmt_sysu.train \
  --config pmt_sysu/config/sysu_pmt.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --pretrained pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  --output outputs/pmt_sysu/official_reproduction \
  --device cuda:0
```

One-batch smoke test:

```bash
python -m pmt_sysu.train \
  --config pmt_sysu/config/sysu_pmt.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --pretrained pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  --output outputs/pmt_sysu/smoke \
  --device cuda:0 \
  --smoke-batches 1
```

Resume:

```bash
python -m pmt_sysu.train \
  --config pmt_sysu/config/sysu_pmt.yaml \
  --resume outputs/pmt_sysu/official_reproduction/checkpoints/latest.pth \
  --device cuda:0
```

## PMT-MBPatch Variant

`pmt_sysu/config/sysu_pmt_mbpatch.yaml` keeps the PMT SYSU data, losses, training schedule, and evaluation protocol unchanged, but replaces the single overlapping patch embedding with a two-branch patch embedding:

```text
[16,16] stride [12,12]
[16,8]  stride [12,6]
```

The first branch is the anchor branch, so the token count remains compatible with the original PMT transformer and losses. ImageNet single-branch patch weights are copied into the anchor branch and resized for the added branch; the 1x1 fusion starts as an anchor-branch identity.

Startup smoke command:

```bash
python -m pmt_sysu.train \
  --config pmt_sysu/config/sysu_pmt_mbpatch.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --pretrained pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  --output outputs/pmt_sysu/mbpatch_smoke \
  --device cuda:0 \
  --smoke-batches 1
```

## Test

Self-trained best checkpoint:

```bash
python -m pmt_sysu.test \
  --config pmt_sysu/config/sysu_pmt.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --weights outputs/pmt_sysu/official_reproduction/checkpoints/best.pth \
  --mode all \
  --gallery-mode single \
  --trials 10 \
  --device cuda:0
```

Official PMT checkpoint:

```bash
python -m pmt_sysu.test \
  --config pmt_sysu/config/sysu_pmt.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --weights pretrained/pmt_sysu_vit_official.pth \
  --mode all \
  --gallery-mode single \
  --trials 10 \
  --device cuda:0
```

Expected official SYSU all-search single-shot reference is approximately Rank-1 67.53%, mAP 64.98%, mINP 51.86%. Reasonable variance is expected; the metric must not be hard-coded.

## Output

```text
outputs/pmt_sysu/<run_name>/
├── config_resolved.yaml
├── train.log
├── metrics.jsonl
├── metrics.csv
├── checkpoints/
│   ├── latest.pth
│   ├── best.pth
│   └── epoch_XX.pth
└── evaluation/
    ├── trial_00.json
    └── average.json
```

## Compatibility Changes

Compared with the official PMT code, this independent version only changes engineering compatibility:

- paths are YAML/CLI driven;
- `.cuda()` is replaced with `.to(device)`;
- `torch.load` uses `map_location`;
- obsolete `Variable` usage is removed;
- old PyTorch `addmm_` calls use the modern signature;
- checkpoint save/resume records optimizer, scaler, best mAP, config, and random states;
- runtime assertions check PMT batch layout and finite losses.

The PMT model, two-stage training schedule, PK sampling layout, MSEL/DCL logic, and SYSU evaluation protocol are preserved.
