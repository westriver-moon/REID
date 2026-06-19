# Commands

Working directory:

```bash
cd /home/cgv841/ybj/TVI-LFM
source /home/cgv841/anaconda3/etc/profile.d/conda.sh
conda activate clipreid
```

Pre-start design check:

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
# Loaded config/stage_a/pmt_vit_stage_a_pmt_recipe_288x144_768.yaml.
# Checked: image-only loader, batch layout, PMT no-projection Identity,
# classifier/text dimensions, frozen text tensors.
PY
```

Forward smoke check:

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
# Checked one full batch for current_epoch=0 and current_epoch=6.
# Both gray_ir and rgb_ir branches produced finite losses.
PY
```

Formal kickoff:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python main.py \
  --config_select config/stage_a/pmt_vit_stage_a_pmt_recipe_288x144_768.yaml \
  > /home/cgv841/ybj/TVI-LFM/train_outputs/stage_a_pmt_recipe_no_projection/a1r_288x144_768_launcher.log 2>&1 &
```

Expected training log:

```text
/home/cgv841/ybj/TVI-LFM/logs/stage_a_pmt_vit_recipe_288x144_768_run1/sysu/Base/Baseline_train[RGB_IR]_pmt_recipe/logs/log.log
```
