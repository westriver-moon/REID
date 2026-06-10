#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="/home/cgv841/.vscode-server/bin"

if [ ! -d "${BASE}" ]; then
  echo "missing ${BASE}" >&2
  exit 1
fi

found=0
for d in "${BASE}"/*; do
  [ -d "${d}/bin" ] || continue
  commit="$(basename "${d}")"
  found=1
  "${SCRIPT_DIR}/install-vscode-code-server-hook.sh" "${commit}"
done

if [ "${found}" -eq 0 ]; then
  echo "no VS Code Server commit directories found" >&2
  exit 1
fi
