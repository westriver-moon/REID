#!/usr/bin/env bash
set -euo pipefail

cd /home/cgv841/ybj/TVI-LFM

GPU="${1:-${CUDA_VISIBLE_DEVICES:-0}}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1

PYTHON=/home/cgv841/anaconda3/envs/clipreid/bin/python
CONFIG=project/sysumm01/configs/sysu_ir_vcm_ir_schp_full_s2_steps800.yaml
RUN_DATE="$(date +%F)"
OUTPUT="${OUTPUT_DIR:-logs/sysu_ir_vcm_ir/schp_full_s2_steps800_ep40_${RUN_DATE}}"

echo "[launcher] $(date '+%F %T') config=${CONFIG} output=${OUTPUT} gpu=${CUDA_VISIBLE_DEVICES}"
"${PYTHON}" project/sysumm01/engine/train.py --config "${CONFIG}" --output "${OUTPUT}"
