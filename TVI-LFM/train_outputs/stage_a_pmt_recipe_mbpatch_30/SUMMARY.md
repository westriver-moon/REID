# Stage A PMT Recipe MBPatch 30-Epoch Run

Goal: launch the multi-patch PMT ViT Stage A experiment on an idle GPU.

Run mode: full kickoff.

Selected configuration:

- `config/stage_a/pmt_vit_stage_a_pmt_recipe_288x144_768_mbpatch.yaml`
- Dataset: SYSU-MM01 at `/home/cgv841/datasets/SYSU-MM01/`
- Image size: `288 x 144`
- Backbone: `PMT_VIT`
- Projection/head dimension: `768`
- Patch embedding:
  - anchor branch: `16 x 16`, stride `12 x 12`
  - second branch: `16 x 8`, stride `12 x 6`
- Training schedule: `30` epochs
- Evaluation: IR query to RGB gallery, every 2 epochs from epoch 2.

Startup status:

- GPU0 was selected because it was idle before launch.
- Training started in tmux session `tvi_lfm_mbpatch_30`.
- Training process PID: `214475`.
- Initial model construction succeeded and the process is running on GPU0.

Main evidence:

- Launcher log: `train_outputs/stage_a_pmt_recipe_mbpatch_30/a1_mbpatch_launcher.log`
- Training log: `logs/stage_a_pmt_vit_recipe_288x144_768_mbpatch_run1/sysu/Base/Baseline_train[RGB_IR]_pmt_recipe/logs/log.log`
- Config snapshot: `logs/stage_a_pmt_vit_recipe_288x144_768_mbpatch_run1/sysu/Base/Baseline_train[RGB_IR]_pmt_recipe/configs.yaml`

