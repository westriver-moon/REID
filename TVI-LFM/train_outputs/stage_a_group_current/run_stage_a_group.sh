#!/usr/bin/env bash
set -euo pipefail

cd /home/cgv841/ybj/TVI-LFM
source /home/cgv841/anaconda3/etc/profile.d/conda.sh
conda activate clipreid
export PYTHONUNBUFFERED=1

stamp() {
  date "+%Y-%m-%d %H:%M:%S %Z"
}

echo "[stage_a] launcher_started $(stamp)"
echo "[stage_a] A0_start $(stamp)"
python main.py --config_select config/stage_a/rn50_ori_stage_a_control.yaml
echo "[stage_a] A0_done $(stamp)"

if [[ "${RUN_A1_AFTER_A0:-0}" != "1" ]]; then
  echo "[stage_a] A1_held $(stamp) set RUN_A1_AFTER_A0=1 to launch A1 manually"
  exit 0
fi

echo "[stage_a] A1_start $(stamp)"
python main.py --config_select config/stage_a/pmt_vit_stage_a.yaml
echo "[stage_a] A1_done $(stamp)"
echo "[stage_a] launcher_done $(stamp)"
