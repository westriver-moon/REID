#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

gpu_id="${1:-0}"

output_a="${OUTPUT_A:-logs/sysumm01/a1_short_calib_180}"
config_a="${CONFIG_A:-project/sysumm01/configs/sysumm01_ir_a1_short_calib_180.yaml}"
CUDA_VISIBLE_DEVICES="${gpu_id}" conda run -n clipreid python -u project/sysumm01/engine/train.py \
  --config "${config_a}" \
  --output "${output_a}" \
  --device cuda \
  --print-freq 50

output_b="${OUTPUT_B:-logs/sysumm01/a1_short_calib_335_lr5e6}"
config_b="${CONFIG_B:-project/sysumm01/configs/sysumm01_ir_a1_short_calib_335_lr5e6.yaml}"
CUDA_VISIBLE_DEVICES="${gpu_id}" conda run -n clipreid python -u project/sysumm01/engine/train.py \
  --config "${config_b}" \
  --output "${output_b}" \
  --device cuda \
  --print-freq 50
