#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/cgv841/ybj/TVI-LFM"
PYTHON_BIN="/home/cgv841/anaconda3/envs/clipreid/bin/python"
LOG_ROOT="${PROJECT_ROOT}/logs/sysu_rgb_ir_align"

declare -a RUNS=(
  "base:0:id_triplet_ep20:${PROJECT_ROOT}/project/sysumm01/configs/rgb_ir_align_base.yaml"
  "align:1:clip_proto_ep20:${PROJECT_ROOT}/project/sysumm01/configs/rgb_ir_align_clip_proto.yaml"
)

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

if [[ "${clean}" -eq 1 ]]; then
  for entry in "${RUNS[@]}"; do
    IFS=':' read -r tag _ name _ <<<"${entry}"
    systemctl --user stop "rgb_ir_align_${tag}.service" >/dev/null 2>&1 || true
    systemctl --user reset-failed "rgb_ir_align_${tag}.service" >/dev/null 2>&1 || true
    rm -rf "${LOG_ROOT}/${name}"
  done
fi

mkdir -p "${LOG_ROOT}"

for entry in "${RUNS[@]}"; do
  IFS=':' read -r tag gpu name config <<<"${entry}"
  output_dir="${LOG_ROOT}/${name}"
  train_log="${output_dir}/train.log"
  service_log="${output_dir}/service.log"
  unit="rgb_ir_align_${tag}"
  mkdir -p "${output_dir}"
  printf '[%s] start unit=%s gpu=%s output=%s\n' "$(date '+%F %T')" "${unit}" "${gpu}" "${output_dir}" | tee -a "${service_log}"
  if [[ "${dry_run}" -eq 1 ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%s %q %q --config %q --output %q --log-file %q\n' \
      "${gpu}" "${PYTHON_BIN}" "${PROJECT_ROOT}/project/sysumm01/engine/train.py" "${config}" "${output_dir}" "${train_log}"
    continue
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
    --log-file "${train_log}"
done

systemctl --user list-units 'rgb_ir_align_*.service' --no-pager || true
