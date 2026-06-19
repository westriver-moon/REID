# Baseline Comparison

All metrics are percentages on SYSU all-search single-shot unless noted.

| Method | Rank-1 | mAP | mINP | Note |
|---|---:|---:|---:|---|
| A0 RN50 | 52.51 | 50.58 | 36.47 | Local Stage A control, TVI-LFM recipe, RGB_IR image-only, stopped at epoch 115; best epoch 107. |
| A1 plain | 24.42 | 24.57 | 13.73 | Naive PMT_VIT replacement under original TVI-LFM wrt,id recipe; best epoch 31/40. |
| PMT recipe 256 | 63.68 | 61.89 | 48.57 | Current TVI-LFM-integrated PMT recipe run2; best/final epoch 23/24. |
| PMT recipe 288 | 64.71 | 62.42 | 48.98 | Current TVI-LFM-integrated PMT recipe run2; best/final epoch 23/24. |
| PMT official | 68.51 | 66.68 | 54.26 | Reference already used in stage_a_group_current comparison; official checkpoint average. |
| PMT mbpatch | 70.44 | 67.66 | 54.86 | Independent PMT-SYSU mbpatch reproduction metrics.csv final epoch 24. |

## Key deltas for current best: A1 PMT recipe 288x144
- vs TVI-LFM A0 RN50 local image-only: Rank-1 +12.21, mAP +11.83, mINP +12.51 points
- vs A1 plain PMT replacement: Rank-1 +40.29, mAP +37.85, mINP +35.26 points
- vs PMT official checkpoint reference: Rank-1 -3.80, mAP -4.26, mINP -5.28 points
- vs PMT-SYSU mbpatch reproduction: Rank-1 -5.73, mAP -5.24, mINP -5.88 points

Plot: `/home/cgv841/ybj/TVI-LFM/train_outputs/stage_a_pmt_recipe_size_compare/plots/baseline_comparison_bars.png`
CSV: `/home/cgv841/ybj/TVI-LFM/train_outputs/stage_a_pmt_recipe_size_compare/baseline_comparison_metrics.csv`
