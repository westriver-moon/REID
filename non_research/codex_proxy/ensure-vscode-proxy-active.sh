#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${SCRIPT_DIR}/auto-heal.log"
EXPECTED_PROXY="ALL_PROXY=socks5h://127.0.0.1:7897"
SERVER_BIN="/home/cgv841/.vscode-server/bin"

"${SCRIPT_DIR}/install-all-vscode-code-server-hooks.sh" >/dev/null 2>&1 || true

if [ ! -d "${SERVER_BIN}" ]; then
  exit 0
fi

bad=0
while IFS= read -r pid; do
  [ -n "${pid}" ] || continue
  if [ ! -r "/proc/${pid}/environ" ]; then
    continue
  fi
  if ! tr '\0' '\n' < "/proc/${pid}/environ" | grep -qx "${EXPECTED_PROXY}"; then
    bad=1
    break
  fi
done < <(pgrep -u cgv841 -f "/home/cgv841/.vscode-server/bin/.*/out/server-main.js" || true)

if [ "${bad}" -eq 1 ]; then
  {
    printf '%s restarting vscode-server because proxy env is missing\n' "$(date '+%Y-%m-%d %H:%M:%S')"
    ps -u cgv841 -eo pid,cmd | grep -F "/home/cgv841/.vscode-server/bin/" | grep -v grep || true
  } >> "${LOG}"
  pkill -u cgv841 -f "/home/cgv841/.vscode-server/bin/" || true
fi
