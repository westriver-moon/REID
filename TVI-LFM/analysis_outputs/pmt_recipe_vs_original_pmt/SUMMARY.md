# TVI-LFM PMT Recipe vs Original PMT-SYSU

This is a read-only comparison of the current TVI-LFM Stage A PMT-recipe runs
against the independent PMT-SYSU implementation in this workspace.

## Current Best Run

- Config: `config/stage_a/pmt_vit_stage_a_pmt_recipe_288x144.yaml`
- Best/final result: Rank-1 64.71, mAP 62.42, mINP 48.98
- Important scope: TVI-LFM integration with a PMT-like recipe, not a pure
  PMT-SYSU reproduction.

## Aligned With PMT-SYSU

- ImageNet ViT-B/16 initialization path is the same family of checkpoint.
- ViT depth, heads, mlp ratio, dropout, attention dropout, drop path, patch
  size, and stride are aligned for the single-branch PMT config.
- PMT recipe transforms are closely aligned: resize, horizontal flip,
  grayscale visible stage, ImageNet normalization, random erasing, and thermal
  ColorJitter/GaussianBlur choice.
- Batch layout is effectively aligned: TVI-LFM uses `batch_size=32`,
  `num_pos=4`, and passes `batch_size / num_pos = 8` identities to the sampler,
  yielding 32 visible and 32 IR samples per step.
- Progressive stage count is aligned in count: six gray/IR epochs followed by
  RGB/IR epochs.
- Triplet, MSEL, and DCL math is copied/ported from PMT-SYSU.

## Main Differences

1. Feature dimension and head are different.
   - PMT-SYSU uses 768-dimensional ViT CLS features, `BatchNorm1d(768)`, and a
     768-to-class classifier.
   - Current TVI-LFM uses raw 768-dimensional PMT tokens projected to 2048
     dimensions, then TVI-LFM `Classifier` with `BatchNorm1d(2048)`.

2. The wrapper is different.
   - PMT-SYSU trains `PMTModel` directly.
   - TVI-LFM trains PMT inside the CLIP2ReID shell. Text modules are frozen for
     image-only mode, but the model structure, parameter names, optimizer
     grouping, checkpoint format, and evaluation path are not PMT-native.

3. BN details differ.
   - PMT-SYSU freezes bottleneck BN bias.
   - TVI-LFM `Classifier` does not freeze BN bias.

4. Current best input size differs from official PMT default.
   - PMT-SYSU default is 256x128.
   - Current best run is 288x144. A 256x128 current run also exists and scored
     lower than 288x144 but still below PMT-SYSU.

5. MBPatch is not implemented in TVI-LFM PMT adapter.
   - The PMT-SYSU `sysu_pmt_mbpatch.yaml` baseline uses a second patch branch
     `[16, 8] / [12, 6]`.
   - TVI-LFM `PMTViTVisual` only constructs the single-branch ViT.

6. LR warmup indexing differs.
   - PMT-SYSU starts epochs at 1, so the first actual epoch uses LR factor
     0.34.
   - TVI-LFM starts at epoch 0, so the first actual epoch uses LR factor 0.01.
   - This makes the TVI-LFM 24-epoch run slightly more conservative early on.

7. Evaluation and checkpoint-selection surfaces differ.
   - TVI-LFM training-time SYSU evaluation averages over 10 gallery trials.
   - PMT-SYSU training loop calls evaluation with `trials=1`, although its test
     config says 10 trials and separate final testing can use 10.
   - TVI-LFM saves best by Rank-1 for IR, while PMT-SYSU tracks best mAP.

## Interpretation

The current TVI-LFM PMT-recipe run should be interpreted as:

> PMT-like training recipe transferred into TVI-LFM/CLIP2ReID infrastructure.

It should not be interpreted as:

> A faithful PMT-SYSU reproduction.

The most likely result-affecting differences are the 768-to-2048 projection,
the 2048-dimensional TVI-LFM head/BN, the non-PMT wrapper and optimizer group
surface, the missing MBPatch branch when comparing to the mbpatch baseline, and
the warmup epoch indexing.
