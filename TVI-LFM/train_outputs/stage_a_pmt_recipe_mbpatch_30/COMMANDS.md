# Commands

GPU check:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits
```

Selected GPU:

```text
GPU0: NVIDIA GeForce RTX 3090, 11 MiB used, 0% utilization, no compute process before launch.
```

Full kickoff command:

```bash
cd /home/cgv841/ybj/TVI-LFM
tmux new-session -d -s tvi_lfm_mbpatch_30 \
  "cd /home/cgv841/ybj/TVI-LFM && source /home/cgv841/anaconda3/etc/profile.d/conda.sh && conda activate clipreid && python main.py --config_select config/stage_a/pmt_vit_stage_a_pmt_recipe_288x144_768_mbpatch.yaml 2>&1 | tee train_outputs/stage_a_pmt_recipe_mbpatch_30/a1_mbpatch_launcher.log"
```

Monitor commands:

```bash
tmux attach -t tvi_lfm_mbpatch_30
tail -f '/home/cgv841/ybj/TVI-LFM/logs/stage_a_pmt_vit_recipe_288x144_768_mbpatch_run1/sysu/Base/Baseline_train[RGB_IR]_pmt_recipe/logs/log.log'
nvidia-smi
```

