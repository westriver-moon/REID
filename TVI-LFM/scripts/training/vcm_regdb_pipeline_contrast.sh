#!/usr/bin/env bash
set -euo pipefail

cd /home/cgv841/ybj/TVI-LFM
PYTHON=${PYTHON:-/home/cgv841/anaconda3/envs/clipreid/bin/python}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="${GPU}"

"${PYTHON}" project/sysumm01/engine/train.py \
  --config project/sysumm01/configs/external_pretrain_vcm_regdb_contrast.yaml

"${PYTHON}" project/sysumm01/engine/train.py \
  --config project/sysumm01/configs/sysu_finetune_from_vcm_regdb_contrast.yaml
