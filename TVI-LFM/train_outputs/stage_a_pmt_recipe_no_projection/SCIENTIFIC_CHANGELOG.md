# Scientific Changelog

Changed:

- Added `config/stage_a/pmt_vit_stage_a_pmt_recipe_288x144_768.yaml`.
- Set `prj_output_dim: 768`, causing PMT visual projection to become identity.
- Set a new output path: `logs/stage_a_pmt_vit_recipe_288x144_768_run1/`.
- Set GPU selection to physical GPU 0.

Held constant against the 2048-dimensional 288x144 PMT-recipe comparison:

- Dataset and SYSU protocol.
- PMT ImageNet ViT-B/16 initialization.
- `288x144` input size.
- Patch size, stride, ViT depth, heads, MLP ratio, dropout, and drop path.
- PMT recipe transforms.
- PMT progressive schedule, Triplet, MSEL, and DCL weights.
- AdamW, LR, weight decay, warmup, cosine schedule, batch size, `num_pos`, seed.
- Image-only training and IR evaluation.

Meaning:

- This isolates the effect of the 768-dimensional PMT-native head versus the
  previous 2048-dimensional TVI-LFM-compatible projected head as much as the
  current codebase allows.
