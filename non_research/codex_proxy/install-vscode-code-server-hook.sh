#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "${1:-}" != "" ]; then
  COMMIT="$1"
else
  COMMIT="$(find /home/cgv841/.vscode-server/bin -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | tail -n 1)"
fi
TARGET="/home/cgv841/.vscode-server/bin/${COMMIT}/bin/code-server"
BACKUP_DIR="${SCRIPT_DIR}/backups"
STAMP="$(date +%Y%m%d-%H%M%S)"
MARKER="CODEX_PROXY_HOOK"

mkdir -p "${BACKUP_DIR}"

if [ -z "${COMMIT}" ]; then
  echo "no vscode-server commit directory found" >&2
  exit 1
fi

if [ ! -f "${TARGET}" ]; then
  echo "missing ${TARGET}" >&2
  exit 1
fi

cp -p "${TARGET}" "${BACKUP_DIR}/code-server.${COMMIT}.${STAMP}.bak"

if grep -q "${MARKER}" "${TARGET}"; then
  echo "hook already present in ${TARGET}"
  exit 0
fi

tmp="${TARGET}.tmp.${STAMP}"
awk '
  {
    print
    if ($0 ~ /^ROOT=/ && inserted != 1) {
      print ""
      print "# CODEX_PROXY_HOOK: source proxy env before VS Code Server starts."
      print "if [ -r /home/cgv841/ybj/non_research/codex_proxy/proxy-env.sh ]; then"
      print "	. /home/cgv841/ybj/non_research/codex_proxy/proxy-env.sh"
      print "	codex_proxy_apply >/dev/null 2>&1 || true"
      print "fi"
      inserted = 1
    }
  }
' "${TARGET}" > "${tmp}"

chmod --reference="${TARGET}" "${tmp}"
mv "${tmp}" "${TARGET}"
echo "installed hook in ${TARGET}"
echo "backup_dir=${BACKUP_DIR}"
