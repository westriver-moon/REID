# Stage A Size Audit Summary

Audit time: 2026-06-19 05:52:06 CST.

Scope:

- `config/stage_a/pmt_vit_stage_a_pmt_recipe_256x128.yaml`
- `config/stage_a/pmt_vit_stage_a_pmt_recipe_288x144.yaml`
- PMT recipe training path in `core/build.py`, `data_loader/loader.py`, `solver/build.py`, `solver/lr_scheduler.py`, and `core/test.py`
- Current run logs under `logs/stage_a_pmt_vit_recipe_*_run2/`

Current judgment:

- The active 256x128 vs 288x144 comparison is internally fair: both resolved configs match except image size, CUDA device/output path, and resulting token grid.
- The loader is image-only for these configs: no training or IR-only evaluation text fields are loaded.
- Batch layout is compatible with PMT losses: visible and IR labels are aligned, and `num_pos` chunks contain one identity.
- Both active runs completed epoch 0 and wrote finite loss values.
- Both active runs completed epoch 1. The 288x144 run already completed the first IR evaluation at epoch 1 / formal epoch 2.
- The current PMT recipe is not a bit-for-bit PMT-SYSU reproduction. It is a TVI-LFM integration that keeps PMT-style progressive training and losses while using CLIP2ReID classifier infrastructure.

Most important caveat:

- `prj_output_dim: 2048` adds a random 768-to-2048 projection before the TVI-LFM classifier/losses. Original PMT-SYSU uses 768-dimensional features. This is acceptable for an internal size ablation because both active runs share it, but it should be disclosed when comparing to PMT-SYSU.
