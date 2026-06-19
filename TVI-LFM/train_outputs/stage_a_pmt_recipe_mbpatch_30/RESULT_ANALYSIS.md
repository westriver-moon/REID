# Stage A MBPatch 30-Epoch Result Analysis

Status: complete. The tmux process has exited, GPU0 is idle, and the log contains epochs `0..29`.

## Best Rank-1 Checkpoint

| Run | Best epoch | Rank-1 | mAP | mINP |
|---|---:|---:|---:|---:|
| 288/768 mbpatch | 27 | 63.85 | 62.65 | 50.40 |
| 288/768 no-proj | 21 | 65.53 | 64.11 | 51.65 |
| 288/2048 proj | 23 | 64.71 | 62.42 | 48.98 |
| 256/2048 proj | 23 | 63.68 | 61.89 | 48.57 |

## Final Evaluation

| Run | Final eval epoch | Rank-1 | mAP | mINP |
|---|---:|---:|---:|---:|
| 288/768 mbpatch | 29 | 63.47 | 62.37 | 50.21 |
| 288/768 no-proj | 23 | 65.44 | 64.11 | 51.58 |
| 288/2048 proj | 23 | 64.71 | 62.42 | 48.98 |
| 256/2048 proj | 23 | 63.68 | 61.89 | 48.57 |

## Key Deltas

Current MBPatch best checkpoint vs `288x144, 768 no-proj`:

- Rank-1: -1.68 points
- mAP: -1.46 points
- mINP: -1.24 points

Current MBPatch best checkpoint vs `288x144, 2048 proj`:

- Rank-1: -0.87 points
- mAP: +0.23 points
- mINP: +1.42 points

Current MBPatch best checkpoint vs `256x128, 2048 proj`:

- Rank-1: +0.17 points
- mAP: +0.76 points
- mINP: +1.83 points

## Interpretation

MBPatch did not improve over the strongest current TVI-LFM-integrated PMT branch, namely `288x144, 768 no-proj`. Its best checkpoint is epoch 27 with Rank-1 63.85%, mAP 62.65%, and mINP 50.40%. The final epoch 29 slightly regresses to Rank-1 63.47%, mAP 62.37%, and mINP 50.21%.

Compared with the older 2048 projection runs, MBPatch is mixed: it is slightly below the `288x144, 2048 proj` run in Rank-1, but above it in mAP and mINP. Compared with `256x128, 2048 proj`, it is modestly better across all three metrics.

The main negative result is therefore specific and useful: adding this two-branch patch fusion on top of the already-best `768 no-proj` design does not help in the current TVI-LFM Stage A setup. The extra 6 epochs did not rescue it; the curve peaks at epoch 27 and then drifts down a little.

## Comparability Notes

This is not a pure single-factor ablation against the previous `288x144, 768 no-proj` run because training length changed from 24 to 30 epochs and the patch embedding changed. The epoch-matched plot at or before epoch 23 is included to show the trajectory under a closer training budget.

## Artifacts

- Dashboard: `/home/cgv841/ybj/TVI-LFM/train_outputs/stage_a_pmt_recipe_mbpatch_30/plots/mbpatch_comparison_dashboard.png`
- Rank-1 curve: `/home/cgv841/ybj/TVI-LFM/train_outputs/stage_a_pmt_recipe_mbpatch_30/plots/mbpatch_eval_rank1.png`
- mAP curve: `/home/cgv841/ybj/TVI-LFM/train_outputs/stage_a_pmt_recipe_mbpatch_30/plots/mbpatch_eval_map.png`
- mINP curve: `/home/cgv841/ybj/TVI-LFM/train_outputs/stage_a_pmt_recipe_mbpatch_30/plots/mbpatch_eval_minp.png`
- Best metric bars: `/home/cgv841/ybj/TVI-LFM/train_outputs/stage_a_pmt_recipe_mbpatch_30/plots/mbpatch_best_metric_bars.png`
- External reference bars: `/home/cgv841/ybj/TVI-LFM/train_outputs/stage_a_pmt_recipe_mbpatch_30/plots/mbpatch_external_reference_bars.png`
- CSV summary: `/home/cgv841/ybj/TVI-LFM/train_outputs/stage_a_pmt_recipe_mbpatch_30/mbpatch_summary_metrics.csv`
- JSON summary: `/home/cgv841/ybj/TVI-LFM/train_outputs/stage_a_pmt_recipe_mbpatch_30/mbpatch_result_summary.json`
