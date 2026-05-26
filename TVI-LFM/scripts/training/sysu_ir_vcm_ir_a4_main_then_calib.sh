#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

gpu_id="${1:-0}"
main_output="${MAIN_OUTPUT:-logs/sysu_ir_vcm_ir/a4_low_ratio_source_aware}"
main_config="${MAIN_CONFIG:-project/sysumm01/configs/sysu_ir_vcm_ir_a4_low_ratio_source_aware.yaml}"
calib_output="${CALIB_OUTPUT:-logs/sysumm01/a4_short_calib_from_a4}"
calib_config="${CALIB_CONFIG:-project/sysumm01/configs/sysumm01_ir_a1_short_calib_335_lr5e6.yaml}"

CUDA_VISIBLE_DEVICES="${gpu_id}" conda run -n clipreid python -u project/sysumm01/engine/train.py \
  --config "${main_config}" \
  --output "${main_output}" \
  --device cuda \
  --print-freq 50

CUDA_VISIBLE_DEVICES="${gpu_id}" conda run -n clipreid python -u project/sysumm01/engine/train.py \
  --config "${calib_config}" \
  --output "${calib_output}" \
  --device cuda \
  --print-freq 50 \
  train.init_checkpoint="${main_output}/checkpoints/best.pth"
