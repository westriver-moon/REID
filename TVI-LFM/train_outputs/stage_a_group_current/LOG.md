# Stage A Training Log

2026-06-18 11:01:23 CST

- Applied startup checks for config, loader, PMT checkpoint, model build, and one-batch backward.
- GPU0 is selected because GPUs 1-3 are occupied by other processes.
- A0 and A1 will run sequentially, not concurrently, to avoid GPU0 memory contention.

Runtime stdout/stderr is captured in `launcher.log`.

2026-06-18 11:01:23 CST

- Background launcher started with PID `2228207`.

2026-06-18 11:02 CST

- PID `2228207` exited immediately because the inline shell launcher had a quoting issue before writing logs.
- Replaced the inline launcher with `run_stage_a_group.sh` and restarted below.
- Background launcher restarted with PID `2229227`.

2026-06-18 11:04 CST

- Foreground `timeout` probe confirmed `run_stage_a_group.sh` reaches A0 data preparation.
- Found and fixed Stage A `output_path` missing trailing slash before full kickoff.
- Full background launcher restarted with PID `2231196` using `setsid`.

2026-06-18 11:07 CST

- Confirmed launcher is alive and running A0: `python main.py --config_select config/stage_a/rn50_ori_stage_a_control.yaml`.
- GPU0 is occupied by the A0 process at about 7.2GB and high utilization.
- A0 training log path: `logs/stage_a_rn50_ori_control/sysu/Base/Baseline_train[RGB_IR]_wrt,id/logs/log.log`.

2026-06-18 11:17 CST

- A0 is still running on GPU0.
- Latest logged epoch: `Epoch: 1`.
- A0 loss trend so far: epoch 0 `total_loss=15.363866`, epoch 1 `total_loss=5.928775`.
- No `Traceback`, `RuntimeError`, or CUDA OOM found in the launcher/A0 log.

2026-06-18 14:03 CST

- A0 is still running.
- Latest logged epoch: `Epoch: 32`; 33/120 epochs have completed.
- Validation has not started because this config begins evaluation after 80 completed epochs and then evaluates every 2 epochs.

2026-06-18 21:19 CST

- A0 is still running as PID `2231211`.
- Latest logged epoch: `Epoch: 114`; 115/120 epochs have completed.
- Best A0 validation so far is from epoch 107: Rank-1 `52.5059%`, mAP `50.5829%`, mINP `36.4707%`.
- Best A0 checkpoint so far: `logs/stage_a_rn50_ori_control/sysu/Base/Baseline_train[RGB_IR]_wrt,id/models/model_IR_107.pth`.
- Parent launcher PID `2231196` was intentionally paused to prevent automatic A1 launch after A0.
- A1 config was shortened to `total_train_epoch: 40`, with `eval_start_epoch: 20` and `eval_epoch: 2`.
- Launcher script now holds A1 unless `RUN_A1_AFTER_A0=1` is explicitly set.

2026-06-18 21:25 CST

- A0 was still running, latest logged epoch `115`.
- Per user request, A0 PID `2231211` and the paused parent launcher PID `2231196` were terminated.
- GPU0 was confirmed free of the A0 process.
- A1 was started manually on GPU0:
  `python main.py --config_select config/stage_a/pmt_vit_stage_a.yaml`.
- A1 launcher PID: `2971520`; A1 training PID: `2971533`.
- A1 log path: `logs/stage_a_pmt_vit/sysu/Base/Baseline_train[RGB_IR]_wrt,id/logs/log.log`.
- A1 startup evidence: `visual_model_name: PMT_VIT`, PMT ImageNet weights loaded, `Missing keys: 0; Unexpected keys: 0`, process using `cuda:0`.

2026-06-19 CST

- A1 completed normally: `[stage_a] A1_manual_done 2026-06-19 04:13:52 CST`.
- A1 logged 40 epochs, from epoch `0` through epoch `39`.
- A1 best validation was epoch 31: Rank-1 `24.4176%`, mAP `24.5673%`, mINP `13.7255%`.
- A1 final validation was epoch 39: Rank-1 `20.9650%`, mAP `21.7375%`, mINP `11.9641%`.
- Generated result artifacts:
  - `stage_a_a0_a1_training_curves.png`
  - `stage_a_a0_a1_validation_curves.png`
  - `stage_a_a0_a1_best_metric_bars.png`
  - `stage_a_reference_comparison_bars.png`
  - `stage_a_results_analysis.md`
