# Stage A Training Summary

Run mode: full kickoff.

Goal: train the Stage A experiment pair:

- A0 control: `RN50_ORI + RGB_IR + wrt,id`
- A1 main: `PMT_VIT + RGB_IR + wrt,id`

Dataset assumption: SYSU-MM01 image arrays are available under `/home/cgv841/datasets/SYSU-MM01/`.

Checkpoint assumption: PMT ImageNet ViT-B/16 checkpoint is available at `/home/cgv841/ybj/PMT-SYSU/pretrained/jx_vit_base_p16_224-80ecf9dd.pth`.

Preflight status before kickoff:

- Stage A configs resolve to `training_mode: RGB_IR`, `joint_mode: image_only`, `test_modality: IR`.
- A0/A1 loaders initialize without reading text assets.
- PMT checkpoint load reports `Missing keys: 0; Unexpected keys: 0`.
- One-batch backward smoke tests pass for A0 and A1 on GPU0.

Current state as of 2026-06-18 21:19 CST:

- A0 is still running on GPU0 under PID `2231211`; latest logged epoch is `114` (`115/120` epochs completed, counting from epoch 0).
- A0 best checkpoint so far is `logs/stage_a_rn50_ori_control/sysu/Base/Baseline_train[RGB_IR]_wrt,id/models/model_IR_107.pth`.
- A0 best validation so far: Rank-1 `52.5059%`, mAP `50.5829%`, mINP `36.4707%`.
- The parent launcher PID `2231196` is intentionally paused so A1 cannot start automatically when A0 exits.
- A1 has been changed to a shorter 40-epoch run with validation from epoch 20 every 2 epochs.

Current state as of 2026-06-18 21:28 CST:

- A0 was stopped by user request at logged epoch `115` before the planned 120-epoch completion.
- The paused parent launcher was also terminated.
- A1 was started manually on GPU0 with PID `2971533` and launcher PID `2971520`.
- A1 startup verified: PMT ImageNet ViT weights loaded with `Missing keys: 0; Unexpected keys: 0`, and the process is using `cuda:0`.

Final state as of 2026-06-19 CST:

- A1 completed the configured 40 epochs.
- A1 best checkpoint: `logs/stage_a_pmt_vit/sysu/Base/Baseline_train[RGB_IR]_wrt,id/models/model_IR_31.pth`.
- A1 best validation at epoch 31: Rank-1 `24.4176%`, mAP `24.5673%`, mINP `13.7255%`.
- A1 final validation at epoch 39: Rank-1 `20.9650%`, mAP `21.7375%`, mINP `11.9641%`.
- Parsed plots and CSV files are in `train_outputs/stage_a_group_current/`; see `stage_a_results_analysis.md`.
