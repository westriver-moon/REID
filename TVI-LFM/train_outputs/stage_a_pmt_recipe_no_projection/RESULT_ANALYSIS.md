# No-Projection Result Analysis

All metrics are SYSU all-search single-shot fractions converted to percentages in the table below.

| Run | Train epochs | Eval points | Best Rank-1 epoch | Best Rank-1 | mAP at Best R1 | mINP at Best R1 | Best mAP epoch | Best mAP | Final eval epoch | Final Rank-1 | Final mAP | Final mINP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 288x144, 768 no-proj | 24 | 12 | 21 | 65.53 | 64.11 | 51.65 | 23 | 64.11 | 23 | 65.44 | 64.11 | 51.58 |
| 288x144, 2048 proj | 24 | 12 | 23 | 64.71 | 62.42 | 48.98 | 23 | 62.42 | 23 | 64.71 | 62.42 | 48.98 |
| 256x128, 2048 proj | 24 | 12 | 23 | 63.68 | 61.89 | 48.57 | 23 | 61.89 | 23 | 63.68 | 61.89 | 48.57 |

## Key Deltas
- 768 no-proj vs 288x144 2048 projection, best-Rank1 checkpoint: rank1 +0.82 pts, mAP_at_best_rank1 +1.69 pts, mINP_at_best_rank1 +2.67 pts
- 768 no-proj vs 288x144 2048 projection, final eval: rank1 +0.73 pts, mAP +1.69 pts, mINP +2.60 pts

## Interpretation
- The 768 no-projection run completed all 24 epochs and slightly outperformed the 2048 projection run at the same 288x144 input size.
- The improvement is modest in Rank-1, stronger in mAP, and largest in mINP, which suggests the native 768-dimensional head improves tail/ranking quality rather than only top-1 retrieval.
- The validation curve keeps improving until the final evaluations, so this result does not indicate obvious overfitting within 24 epochs.
- Because this is still TVI-LFM/CLIP2ReID infrastructure, it is evidence against the 2048 projection being necessary for Stage A, not a pure PMT-SYSU reproduction claim.

## Artifacts
- `plots/projection_comparison_dashboard.png`
- `plots/projection_eval_rank1.png`
- `plots/projection_eval_map.png`
- `plots/projection_eval_minp.png`
- `plots/projection_train_total_loss.png`
- `plots/projection_train_acc.png`
- `plots/projection_best_metric_bars.png`
- `no_projection_train_metrics.csv`
- `no_projection_eval_metrics.csv`
- `no_projection_summary_metrics.csv`
- `no_projection_result_summary.json`
