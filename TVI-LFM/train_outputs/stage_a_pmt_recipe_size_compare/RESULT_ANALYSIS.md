# Stage A PMT Recipe Size Comparison Results

Completed runs:

- `A1R_256x128`: started `2026-06-19 05:42:09 CST`, finished `2026-06-19 08:12:48 CST`.
- `A1R_288x144`: started `2026-06-19 05:42:09 CST`, finished `2026-06-19 07:48:16 CST`.

Both runs completed 24 epochs and 12 IR evaluations.

## Best IR Evaluation

| Run | Best epoch | Rank-1 | mAP | mINP |
|---|---:|---:|---:|---:|
| 256x128 | 23 | 63.68 | 61.89 | 48.57 |
| 288x144 | 23 | 64.71 | 62.42 | 48.98 |
| 288 - 256 | - | +1.03 | +0.53 | +0.41 |

## Final Training State

| Run | Final epoch | Total loss | ID loss | Triplet loss | MSEL | DCL | Train acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256x128 | 23 | 0.3091 | 0.0517 | 0.0263 | 0.0215 | 0.2097 | 99.27 |
| 288x144 | 23 | 0.3197 | 0.0586 | 0.0270 | 0.0214 | 0.2127 | 99.16 |

## Interpretation

- The larger `288x144` input wins, but the margin is modest: about `+1.03` Rank-1, `+0.53` mAP, and `+0.41` mINP.
- Training curves are almost overlapped, so the difference is unlikely to come from one run failing to optimize.
- Evaluation curves continue improving through the last evaluation. This suggests 24 epochs is near a plateau but not clearly past it.
- The larger input likely helps by preserving more spatial detail and using a larger PMT token grid, but the gain is not large enough to treat as decisive without seed repetition.
- The comparison remains internally fair because optimizer, schedule, seed, batch layout, recipe losses, and evaluation protocol are held fixed.

Important caveat: these are TVI-LFM PMT-adapter runs with a `768 -> 2048` projection/head, not a pure PMT-SYSU 768-dim reproduction.

## Artifacts

- Training metrics: `stage_a_size_train_metrics.csv`
- Evaluation metrics: `stage_a_size_eval_metrics.csv`
- Parsed summary: `stage_a_size_result_summary.json`
- Plots:
  - `plots/training_dashboard.png`
  - `plots/eval_dashboard.png`
  - `plots/eval_rank1.png`
  - `plots/eval_map.png`
  - `plots/eval_minp.png`
