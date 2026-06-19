# Comparability Report

Strongest comparison target:

- Completed `288x144 / 768 no-projection` Stage A PMT recipe run.

Single-factor interpretation is approximate, not perfect:

- The architecture changed through multi-patch branch fusion.
- The training length changed from 24 to 30 epochs.

Recommended analysis after completion:

- Compare best Rank-1, mAP, and mINP against:
  - `288x144 / 768 no-proj`
  - `288x144 / 2048 projection`
  - `256x128 / 2048 projection`
- Inspect whether gains come from later epochs or from the multi-patch branch by plotting epoch-wise curves.
- If this run improves, a cleaner follow-up ablation would be `288x144 / 768 mbpatch / 24 epochs`.

