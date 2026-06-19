# Stage A Training Commands

Started from repository root:

```bash
cd /home/cgv841/ybj/TVI-LFM
source /home/cgv841/anaconda3/etc/profile.d/conda.sh
conda activate clipreid

python main.py --config_select config/stage_a/rn50_ori_stage_a_control.yaml
python main.py --config_select config/stage_a/pmt_vit_stage_a.yaml
```

Launcher log:

```text
/home/cgv841/ybj/TVI-LFM/train_outputs/stage_a_group_current/launcher.log
```

The original kickoff ran commands sequentially on GPU0 through one background launcher.

Runtime control update on 2026-06-18 21:19 CST:

- The parent launcher PID `2231196` was paused while A0 PID `2231211` continues running, so A1 will not auto-start after A0.
- `run_stage_a_group.sh` now requires `RUN_A1_AFTER_A0=1` before it will launch A1.
- A1 should be launched manually after reviewing A0 completion, using the 40-epoch config now stored in `config/stage_a/pmt_vit_stage_a.yaml`.

Manual A1 launch on 2026-06-18 21:25 CST:

```bash
cd /home/cgv841/ybj/TVI-LFM
source /home/cgv841/anaconda3/etc/profile.d/conda.sh
conda activate clipreid
python main.py --config_select config/stage_a/pmt_vit_stage_a.yaml
```

A1 launcher log:

```text
/home/cgv841/ybj/TVI-LFM/train_outputs/stage_a_group_current/a1_launcher.log
```

Actual launcher script:

```text
/home/cgv841/ybj/TVI-LFM/train_outputs/stage_a_group_current/run_stage_a_group.sh
```
