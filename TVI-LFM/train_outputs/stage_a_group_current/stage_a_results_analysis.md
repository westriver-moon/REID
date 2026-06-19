# Stage A A0/A1 Result Analysis

Generated: 2026-06-19 CST.

## Artifacts

- `stage_a_a0_a1_training_curves.png`: training loss and accuracy curves.
- `stage_a_a0_a1_validation_curves.png`: validation Rank-1, mAP, and mINP curves.
- `stage_a_a0_a1_best_metric_bars.png`: A0 vs A1 best validation metrics.
- `stage_a_reference_comparison_bars.png`: A0/A1 vs local PMT official checkpoint reference, not strict comparability.
- `stage_a_a0_a1_train_curves.csv`, `stage_a_a0_a1_eval_curves.csv`, `stage_a_a0_a1_summary_metrics.csv`: parsed numeric data.

## Summary

| Run | Train epochs logged | Eval points | Best epoch | Best Rank-1 | Best mAP | Best mINP | Final eval epoch | Final Rank-1 | Final mAP | Final mINP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0_RN50_ORI | 116 | 18 | 107 | 52.51% | 50.58% | 36.47% | 113 | 52.15% | 50.23% | 36.18% |
| A1_PMT_VIT | 40 | 11 | 31 | 24.42% | 24.57% | 13.73% | 39 | 20.97% | 21.74% | 11.96% |

## Main Observations

- A1 PMT_VIT best validation appears at epoch 31: Rank-1 24.42%, mAP 24.57%, mINP 13.73%.
- A1 final eval at epoch 39 falls to Rank-1 20.97%, mAP 21.74%, mINP 11.96%.
- A1 final Rank-1 is 3.45 percentage points below its own best, while training loss keeps decreasing to 1.0182; this suggests validation degradation after epoch 31.
- A0 best validation remains much stronger: A1 best is 28.09 Rank-1 points, 26.02 mAP points, and 22.75 mINP points below A0 best.
- A1 last-3 validation average is Rank-1 21.22%, mAP 22.11%, so the late-stage behavior is not a stable climb.

## Comparability Notes

- A0 and A1 are no longer equal-budget runs: A0 was stopped after logged epoch 115; A1 was intentionally shortened to 40 epochs.
- A1 also inherited `milestones=(40,60,100)`, so the 40-epoch run effectively never enters the first learning-rate decay stage; it only has warmup plus base-LR training.
- Therefore the result is strong evidence that this Stage A PMT insertion is currently underperforming, but it is not yet a fair final verdict on PMT as a backbone under a tuned schedule.
- Local PMT official checkpoint reference reports average Rank-1 68.51%, mAP 66.68%, mINP 54.26%; that run uses PMT's own trained checkpoint/protocol and is not directly comparable to this ImageNet-initialized TVI-LFM insertion.

## Interpretation

The main failure signal is not that PMT_VIT cannot reduce training loss; it can. The issue is that identity accuracy and validation retrieval quality remain far below the RN50 control, and validation peaks early around epoch 31 before dropping. This points to a recipe/integration mismatch: the PMT visual backbone is being trained inside TVI-LFM with the original RN50-oriented optimizer and schedule, no PMT SYSU checkpoint, no PMT losses, and no schedule retuning for the shortened 40-epoch budget.

Most likely next checks: retune A1 learning rate/schedule for 40 epochs, test a smaller visual LR for PMT, and run an A1 variant initialized from a SYSU-trained PMT checkpoint if the goal is to compare architecture rather than ImageNet-only initialization.
