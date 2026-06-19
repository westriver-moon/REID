# Scientific Changelog

This run differs from the completed `288x144 / 768 no-projection` PMT recipe run by adding a multi-patch visual patch embedding and increasing the schedule from 24 to 30 epochs.

Changed:

- Enabled two-branch PMT patch embedding.
- Added a second branch with `16 x 8` patches and `12 x 6` stride.
- Fused branch feature maps with a `1 x 1` convolution before tokenization.
- Set `total_train_epoch: 30`.

Held constant against the previous closest Stage A PMT recipe run:

- SYSU-MM01 dataset path and protocol.
- `288 x 144` image size.
- `PMT_VIT` ViT-B style backbone.
- ImageNet ViT-B/16 checkpoint initialization.
- `prj_output_dim: 768`.
- `pmt_recipe` losses and progressive schedule.
- IR-only evaluation.

Interpretation boundary:

This is not a pure PMT-SYSU reproduction. It remains a TVI-LFM Stage A integration with a PMT-style backbone and training recipe.

