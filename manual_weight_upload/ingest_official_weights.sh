#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/cgv841/ybj"
UPLOAD_DIR="$ROOT_DIR/manual_weight_upload/inbox"
DEST_DIR="$ROOT_DIR/pretrained/official"
LOG_DIR="$ROOT_DIR/manual_weight_upload/logs"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/ingest_${TS}.log"

RGB_FILE="transreid_market_official.pth"
IR_FILE="vitb16_ics_official.pth"
MIN_BYTES=$((100 * 1024 * 1024))

mkdir -p "$UPLOAD_DIR" "$DEST_DIR" "$LOG_DIR"

log() {
  echo "[$(date +'%F %T')] $*" | tee -a "$LOG_FILE"
}

check_one() {
  local path="$1"
  local name="$2"

  if [[ ! -f "$path" ]]; then
    log "ERROR: missing file: $name"
    return 1
  fi

  local size
  size=$(stat -c '%s' "$path")
  if [[ "$size" -lt "$MIN_BYTES" ]]; then
    log "ERROR: file too small ($size bytes): $name"
    log "HINT: this may be an HTML page or incomplete download"
    return 1
  fi

  local sha
  sha=$(sha256sum "$path" | awk '{print $1}')
  log "OK: $name size=$size sha256=$sha"
  return 0
}

copy_one() {
  local src="$1"
  local dst="$2"
  cp -f "$src" "$dst"
  sync
  log "COPIED: $src -> $dst"
}

main() {
  log "Start ingest workflow"
  log "UPLOAD_DIR=$UPLOAD_DIR"
  log "DEST_DIR=$DEST_DIR"

  local rgb_src="$UPLOAD_DIR/$RGB_FILE"
  local ir_src="$UPLOAD_DIR/$IR_FILE"

  check_one "$rgb_src" "$RGB_FILE"
  check_one "$ir_src" "$IR_FILE"

  copy_one "$rgb_src" "$DEST_DIR/$RGB_FILE"
  copy_one "$ir_src" "$DEST_DIR/$IR_FILE"

  log "SUCCESS: both official weights are ingested"
  log "DEST RGB: $DEST_DIR/$RGB_FILE"
  log "DEST IR : $DEST_DIR/$IR_FILE"
}

main "$@"
