#!/usr/bin/env bash
set -euo pipefail

cd /home/cgv841/ybj/TVI-LFM

GPU="${1:-${CUDA_VISIBLE_DEVICES:-0}}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1

PYTHON=/home/cgv841/anaconda3/envs/clipreid/bin/python
CONFIG=project/sysumm01/configs/sysu_finetune_from_vcm_regdb_no_contrast.yaml

echo "[launcher] $(date '+%F %T') config=${CONFIG} gpu=${CUDA_VISIBLE_DEVICES}"
"${PYTHON}" project/sysumm01/engine/train.py --config "${CONFIG}"

