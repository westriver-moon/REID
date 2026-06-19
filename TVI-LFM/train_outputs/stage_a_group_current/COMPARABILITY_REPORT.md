# Comparability Report

A0 and A1 are intended to be compared only against each other as a Stage A pair.

Comparable settings:

- Same dataset: SYSU-MM01
- Same input size: `288 x 144`
- Same batch size: `32`
- Same `num_pos`: `4`
- Same seed: `1`
- Same training mode and losses: `RGB_IR`, `wrt,id`
- Same IR evaluation modality

Non-comparable settings:

- After 2026-06-18 21:19 CST, A1 is configured for 40 epochs while A0 remains a 120-epoch run; this makes A1 a shorter-budget diagnostic run rather than a strict equal-epoch comparison.
- Do not compare A1 directly to older historical RN50 numbers trained with text/fusion or different recipes.
- Do not compare this run to PMT-SYSU runs using MSEL, DCL, progressive training, or official PMT SYSU checkpoint initialization.
