# Scientific Changelog

Change type: architecture variant.

Baseline preserved:

- Dataset remains SYSU-MM01 RGB-IR.
- PMT sampler and RGB/IR label-aligned batch layout remain unchanged.
- PMT two-stage schedule remains unchanged.
- PMT ID, Triplet, MSEL, and DCL losses remain unchanged.
- SYSU all-search single-shot evaluation protocol remains unchanged.

Changed:

- The ViT patch embedding can now be configured as multi-branch.
- The MBPatch config uses an anchor branch `[16,16] stride [12,12]` plus an added branch `[16,8] stride [12,6]`.
- Branch feature maps are resized to the anchor grid, concatenated, and fused with a 1x1 convolution back to 768 channels.

Research status:

- This is a candidate architecture improvement.
- Startup verification passed.
- No full training or final 10-trial evaluation has been run for this variant yet.

