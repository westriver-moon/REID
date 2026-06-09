#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/cgv841/ybj/TVI-LFM"
PYTHON_BIN="/home/cgv841/anaconda3/envs/clipreid/bin/python"
LOG_ROOT="${PROJECT_ROOT}/logs/ablation_patch"

declare -a RUNS=(
  "A:0:A_vit224_square:/home/cgv841/ybj/TVI-LFM/project/sysumm01/configs/ablation_patch/A_vit224_square.yaml"
  "B:1:B_multibranch_288x144:/home/cgv841/ybj/TVI-LFM/project/sysumm01/configs/ablation_patch/B_multibranch_288x144.yaml"
  "C:2:C_singlebranch_288x144:/home/cgv841/ybj/TVI-LFM/project/sysumm01/configs/ablation_patch/C_singlebranch_288x144.yaml"
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
    systemctl --user stop "ablation_patch_${tag}.service" >/dev/null 2>&1 || true
    systemctl --user reset-failed "ablation_patch_${tag}.service" >/dev/null 2>&1 || true
    rm -rf "${LOG_ROOT}/${name}"
  done
fi

mkdir -p "${LOG_ROOT}"

for entry in "${RUNS[@]}"; do
  IFS=':' read -r tag gpu name config <<<"${entry}"
  output_dir="${LOG_ROOT}/${name}"
  mkdir -p "${output_dir}"
  unit="ablation_patch_${tag}"
  train_log="${output_dir}/train.log"
  service_log="${output_dir}/service.log"
  cmd=(
    "${PYTHON_BIN}"
    "${PROJECT_ROOT}/project/sysumm01/engine/train.py"
    --config "${config}"
    --output "${output_dir}"
    --log-file "${train_log}"
  )

  printf '[%s] start unit=%s gpu=%s output=%s\n' "$(date '+%F %T')" "${unit}" "${gpu}" "${output_dir}" | tee -a "${service_log}"

  if [[ "${dry_run}" -eq 1 ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%s %q' "${gpu}" "${cmd[0]}"
    printf ' %q' "${cmd[@]:1}"
    printf '\n'
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
    "${cmd[@]}"
done

systemctl --user list-units 'ablation_patch_*.service' --no-pager
