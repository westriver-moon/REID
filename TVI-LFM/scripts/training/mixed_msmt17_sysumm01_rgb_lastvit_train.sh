#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

gpu_id="${1:-0}"
msmt_root="${MSMT17_ROOT:-/home/cgv841/datasets/MSMT17_V1}"
output="${OUTPUT_DIR:-logs/mixed_msmt17_sysumm01/rgb_lastvit_ep40}"
config="project/sysumm01/configs/mixed_msmt17_sysumm01_rgb_lastvit.yaml"

python - <<PY
from pathlib import Path
config = Path("${config}")
text = config.read_text()
text = text.replace("msmt_root: /home/cgv841/datasets/MSMT17_V1", "msmt_root: ${msmt_root}")
tmp = Path("${output}") / "runtime_config.yaml"
tmp.parent.mkdir(parents=True, exist_ok=True)
tmp.write_text(text)
print(tmp)
PY

runtime_config="${output}/runtime_config.yaml"

CUDA_VISIBLE_DEVICES="${gpu_id}" conda run -n clipreid python -u project/sysumm01/engine/train.py \
  --config "${runtime_config}" \
  --output "${output}" \
  --device cuda \
  --print-freq 50
