# Comparability Report

The PMT-MBPatch variant is directly comparable to the current PMT-SYSU baseline if trained and evaluated with:

- the same SYSU-MM01 data root;
- the same ImageNet ViT initialization;
- the same 24 epoch training schedule;
- the same all-search single-shot 10-trial evaluation.

What remains comparable:

- Data split and evaluation protocol.
- Loss definitions.
- Batch layout.
- Optimizer family and schedule.

What changes:

- Backbone patch embedding capacity increases.
- Parameter count increases to `87,590,400` total parameters.
- ImageNet pretraining is not strict-identical because newly added branch and fusion parameters have no direct original checkpoint keys.

Current evidence supports only startup correctness, not metric improvement.

