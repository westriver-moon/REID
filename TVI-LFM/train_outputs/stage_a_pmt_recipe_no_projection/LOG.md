# Log

- Created no-projection config derived from the existing 288x144 PMT-recipe run.
- Confirmed physical GPU 0 was idle before launch; GPUs 1-3 were busy.
- Base Python lacked `torch`; selected the `clipreid` conda environment used by previous TVI-LFM Stage A runs.
- Pre-start config/model/loader check passed.
- Forward smoke check passed for both progressive branches.
- A first ordinary `nohup` launch exited without keeping a process alive under this tool runner and wrote no output.
- A 60-second foreground startup window confirmed the training command reaches optimizer preparation.
- Formal training was relaunched with `setsid` so it survives the tool session.
- 2026-06-19 12:43:56 CST: formal process `4066067` is running under session `4066067`, parent `1`, on GPU 0. No completed epoch has been logged yet.
- 2026-06-19 12:44:19 CST: epoch 0 completed. First logged metrics: `id_loss=11.939231`, `tri_loss=7.6401796`, `msel_loss=0.0`, `dcl_loss=0.0`, `total_loss=19.579418`, `acc=0.009720726`.
- 2026-06-19 15:40:32 CST: final validation completed at epoch 23.
- Best Rank-1 checkpoint: epoch 21, Rank-1 `65.53%`, mAP `64.11%`, mINP `51.65%`.
- Best mAP / final validation: epoch 23, Rank-1 `65.44%`, mAP `64.11%`, mINP `51.58%`.
- Compared with the matched 288x144 2048-projection run, 768 no-projection improved best Rank-1 by `+0.82`, mAP-at-best-R1 by `+1.69`, and mINP-at-best-R1 by `+2.67` percentage points.
