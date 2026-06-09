#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/cgv841/ybj/TVI-LFM"
PYTHON_BIN="/home/cgv841/anaconda3/envs/clipreid/bin/python"
LOG_ROOT="${PROJECT_ROOT}/logs/ablation_patch_seed"
B_CONFIG="${PROJECT_ROOT}/project/sysumm01/configs/ablation_patch/B_multibranch_288x144.yaml"
C_CONFIG="${PROJECT_ROOT}/project/sysumm01/configs/ablation_patch/C_singlebranch_288x144.yaml"

usage() {
  printf 'Usage: %s [--clean] [--dry-run]\n' "$(basename "$0")"
}

clean=0
dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      clean=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

declare -a UNITS=(
  "ablation_patch_B_seed43"
  "ablation_patch_C_seed43"
  "ablation_patch_seed44_pair"
)

if [[ "${clean}" -eq 1 ]]; then
  for unit in "${UNITS[@]}"; do
    systemctl --user stop "${unit}.service" >/dev/null 2>&1 || true
    systemctl --user reset-failed "${unit}.service" >/dev/null 2>&1 || true
  done
  rm -rf \
    "${LOG_ROOT}/B_seed43" \
    "${LOG_ROOT}/C_seed43" \
    "${LOG_ROOT}/B_seed44" \
    "${LOG_ROOT}/C_seed44"
fi

mkdir -p "${LOG_ROOT}"

start_single() {
  local unit="$1"
  local gpu="$2"
  local name="$3"
  local config="$4"
  local seed="$5"
  local output_dir="${LOG_ROOT}/${name}"
  local service_log="${output_dir}/service.log"
  local train_log="${output_dir}/train.log"
  mkdir -p "${output_dir}"
  printf '[%s] start unit=%s gpu=%s seed=%s output=%s\n' "$(date '+%F %T')" "${unit}" "${gpu}" "${seed}" "${output_dir}" | tee -a "${service_log}"
  if [[ "${dry_run}" -eq 1 ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%s %q %q --config %q --output %q --log-file %q --seed %q\n' \
      "${gpu}" "${PYTHON_BIN}" "${PROJECT_ROOT}/project/sysumm01/engine/train.py" "${config}" "${output_dir}" "${train_log}" "${seed}"
    return
  fi
  systemd-run --user \
    --unit="${unit}" \
    --same-dir \
    --collect \
    --property=WorkingDirectory="${PROJECT_ROOT}" \
    --property=StandardOutput="append:${service_log}" \
    --property=StandardError="append:${service_log}" \
    --setenv="CUDA_VISIBLE_DEVICES=${gpu}" \
    --setenv="PYTHONUNBUFFERED=1" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/project/sysumm01/engine/train.py" \
    --config "${config}" \
    --output "${output_dir}" \
    --log-file "${train_log}" \
    --seed "${seed}"
}

start_seed44_pair() {
  local gpu="3"
  local unit="ablation_patch_seed44_pair"
  local service_dir="${LOG_ROOT}/seed44_pair"
  local service_log="${service_dir}/service.log"
  local b_output="${LOG_ROOT}/B_seed44"
  local c_output="${LOG_ROOT}/C_seed44"
  mkdir -p "${service_dir}" "${b_output}" "${c_output}"
  printf '[%s] start unit=%s gpu=%s seed=44 outputs=%s,%s\n' "$(date '+%F %T')" "${unit}" "${gpu}" "${b_output}" "${c_output}" | tee -a "${service_log}"
  if [[ "${dry_run}" -eq 1 ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%s sequential B_seed44 then C_seed44\n' "${gpu}"
    return
  fi
  systemd-run --user \
    --unit="${unit}" \
    --same-dir \
    --collect \
    --property=WorkingDirectory="${PROJECT_ROOT}" \
    --property=StandardOutput="append:${service_log}" \
    --property=StandardError="append:${service_log}" \
    --setenv="CUDA_VISIBLE_DEVICES=${gpu}" \
    --setenv="PYTHONUNBUFFERED=1" \
    /usr/bin/env bash -lc "\
      set -euo pipefail; \
      '${PYTHON_BIN}' '${PROJECT_ROOT}/project/sysumm01/engine/train.py' --config '${B_CONFIG}' --output '${b_output}' --log-file '${b_output}/train.log' --seed 44; \
      '${PYTHON_BIN}' '${PROJECT_ROOT}/project/sysumm01/engine/train.py' --config '${C_CONFIG}' --output '${c_output}' --log-file '${c_output}/train.log' --seed 44"
}

start_single "ablation_patch_B_seed43" "0" "B_seed43" "${B_CONFIG}" "43"
start_single "ablation_patch_C_seed43" "1" "C_seed43" "${C_CONFIG}" "43"
start_seed44_pair

systemctl --user list-units 'ablation_patch*_seed*.service' --no-pager || true
