#!/usr/bin/env bash
set -euo pipefail

tmp="$(mktemp)"
crontab -l 2>/dev/null \
  | grep -v "install-all-vscode-code-server-hooks.sh" \
  | grep -v "ensure-vscode-proxy-active.sh" > "${tmp}" || true

printf '%s\n' '*/1 * * * * /home/cgv841/ybj/non_research/codex_proxy/ensure-vscode-proxy-active.sh >/dev/null 2>&1' >> "${tmp}"
crontab "${tmp}"
rm -f "${tmp}"

crontab -l | grep -E "ensure-vscode-proxy-active|install-all-vscode" || true
