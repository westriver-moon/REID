# Risks

1. High: projection/head mismatch.
   - Current TVI-LFM optimizes and evaluates a 2048-dimensional projected
     feature, while PMT-SYSU optimizes and evaluates the original 768-dimensional
     ViT feature after BN. This can easily explain part of the remaining gap.

2. High: comparing against PMT-SYSU mbpatch is not apples-to-apples.
   - The mbpatch config adds a second patch branch. TVI-LFM's PMT adapter does
     not implement this branch, so the 70.44/67.66/54.86 reference is a stronger
     architecture than the current adapter.

3. Medium: warmup starts one epoch colder in TVI-LFM.
   - TVI-LFM epoch 0 uses LR factor 0.01, whereas PMT-SYSU epoch 1 uses 0.34.
     This reduces effective early training within a short 24-epoch budget.

4. Medium: BN bias behavior differs.
   - PMT-SYSU freezes bottleneck BN bias. TVI-LFM does not freeze classifier BN
     bias. This is small but affects exact reproduction.

5. Medium: evaluation references mix trial counts.
   - TVI-LFM training-time result is 10-trial averaged. PMT-SYSU training
     metrics are often one-trial, while official checkpoint references may be
     10-trial. These should be labeled carefully in tables.

6. Low: image size difference is not the primary gap.
   - TVI-LFM 288x144 beats TVI-LFM 256x128 by about 1 Rank-1 point, but remains
     below PMT references. Size helps, but it is unlikely to be the main cause.
