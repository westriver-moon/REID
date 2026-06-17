#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/utils/wait_gpu_and_train.sh [options] -- <training command>

Options:
  --gpu ID|auto          GPU id to wait for, or auto-select the first idle GPU. Default: auto
  --max-util PERCENT     GPU utilization must be <= this value. Default: 10
  --max-mem-mb MB        Used GPU memory must be <= this value. Default: 1000
  --max-procs COUNT      Compute process count must be <= this value. Default: 0
  --interval SEC         Seconds between checks. Default: 60
  --stable-checks N      Required consecutive passing checks. Default: 3
  --log FILE             Append training output to FILE.
  --dry-run              Only wait and print the command; do not start training.
  -h, --help             Show this help.

Examples:
  scripts/utils/wait_gpu_and_train.sh --gpu auto --log logs/auto_train.log -- \
    bash scripts/training/regdb_pair_clip_stage2_train.sh 1

  scripts/utils/wait_gpu_and_train.sh --gpu 1 --max-mem-mb 2000 -- \
    bash scripts/training/mixed_msmt17_sysumm01_rgb_lastvit_train.sh '{gpu}'

Notes:
  Use {gpu} in the training command when the wrapped script needs the selected
  physical GPU id as an argument.
EOF
}

gpu="auto"
max_util=10
max_mem_mb=1000
max_procs=0
interval=60
stable_checks=3
log_file=""
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      gpu="${2:?missing value for --gpu}"
      shift 2
      ;;
    --max-util)
      max_util="${2:?missing value for --max-util}"
      shift 2
      ;;
    --max-mem-mb)
      max_mem_mb="${2:?missing value for --max-mem-mb}"
      shift 2
      ;;
    --max-procs)
      max_procs="${2:?missing value for --max-procs}"
      shift 2
      ;;
    --interval)
      interval="${2:?missing value for --interval}"
      shift 2
      ;;
    --stable-checks)
      stable_checks="${2:?missing value for --stable-checks}"
      shift 2
      ;;
    --log)
      log_file="${2:?missing value for --log}"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "Missing training command after --" >&2
  usage >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. This script needs NVIDIA driver tools." >&2
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../.." && pwd)"
cd "${project_root}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

gpu_count() {
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l | tr -d ' '
}

gpu_uuid() {
  local id="$1"
  nvidia-smi --id="${id}" --query-gpu=uuid --format=csv,noheader,nounits | tr -d ' '
}

compute_proc_count() {
  local id="$1"
  local uuid
  uuid="$(gpu_uuid "${id}")"
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' -v uuid="${uuid}" '
        {
          gsub(/^[ \t]+|[ \t]+$/, "", $1)
          if ($1 == uuid) count++
        }
        END { print count + 0 }
      '
}

gpu_snapshot() {
  local id="$1"
  nvidia-smi --id="${id}" \
    --query-gpu=utilization.gpu,memory.used,memory.total,name \
    --format=csv,noheader,nounits
}

gpu_passes() {
  local id="$1"
  local snapshot util mem_used mem_total name procs
  snapshot="$(gpu_snapshot "${id}")"
  IFS=',' read -r util mem_used mem_total name <<<"${snapshot}"
  util="$(echo "${util}" | tr -d ' ')"
  mem_used="$(echo "${mem_used}" | tr -d ' ')"
  mem_total="$(echo "${mem_total}" | tr -d ' ')"
  name="$(echo "${name}" | sed 's/^[[:space:]]*//')"
  procs="$(compute_proc_count "${id}")"

  printf '[%s] gpu=%s util=%s%% mem=%s/%sMB procs=%s name=%s\n' \
    "$(timestamp)" "${id}" "${util}" "${mem_used}" "${mem_total}" "${procs}" "${name}" >&2

  [[ "${util}" -le "${max_util}" && "${mem_used}" -le "${max_mem_mb}" && "${procs}" -le "${max_procs}" ]]
}

pick_gpu_once() {
  local count id
  count="$(gpu_count)"
  for ((id = 0; id < count; id++)); do
    if gpu_passes "${id}"; then
      echo "${id}"
      return 0
    fi
  done
  return 1
}

chosen_gpu=""
passes=0

echo "[$(timestamp)] Waiting for GPU: gpu=${gpu}, max_util=${max_util}%, max_mem=${max_mem_mb}MB, max_procs=${max_procs}, stable_checks=${stable_checks}" >&2

while [[ "${passes}" -lt "${stable_checks}" ]]; do
  if [[ "${gpu}" == "auto" ]]; then
    if candidate="$(pick_gpu_once)"; then
      if [[ -z "${chosen_gpu}" || "${chosen_gpu}" == "${candidate}" ]]; then
        chosen_gpu="${candidate}"
        passes=$((passes + 1))
      else
        chosen_gpu="${candidate}"
        passes=1
      fi
      echo "[$(timestamp)] Candidate GPU ${chosen_gpu} passed ${passes}/${stable_checks} checks." >&2
    else
      chosen_gpu=""
      passes=0
      echo "[$(timestamp)] No GPU is idle enough yet." >&2
    fi
  else
    chosen_gpu="${gpu}"
    if gpu_passes "${chosen_gpu}"; then
      passes=$((passes + 1))
      echo "[$(timestamp)] GPU ${chosen_gpu} passed ${passes}/${stable_checks} checks." >&2
    else
      passes=0
      echo "[$(timestamp)] GPU ${chosen_gpu} is busy." >&2
    fi
  fi

  if [[ "${passes}" -lt "${stable_checks}" ]]; then
    sleep "${interval}"
  fi
done

cmd=("$@")
for i in "${!cmd[@]}"; do
  cmd[$i]="${cmd[$i]//\{gpu\}/${chosen_gpu}}"
done

echo "[$(timestamp)] Starting training on GPU ${chosen_gpu}: ${cmd[*]}" >&2

if [[ "${dry_run}" -eq 1 ]]; then
  echo "[$(timestamp)] Dry run enabled; training was not started." >&2
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${chosen_gpu}"

if [[ -n "${log_file}" ]]; then
  mkdir -p "$(dirname "${log_file}")"
  exec "${cmd[@]}" >>"${log_file}" 2>&1
else
  exec "${cmd[@]}"
fi
