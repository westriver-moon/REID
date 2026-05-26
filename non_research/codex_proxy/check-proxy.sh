#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=proxy-env.sh
. "${SCRIPT_DIR}/proxy-env.sh"

echo "== Codex proxy status =="
if codex_proxy_apply; then
  echo "proxy_env=enabled"
else
  echo "proxy_env=disabled"
fi
codex_proxy_status

echo
echo "== Listening sockets matching ${CODEX_PROXY_PORT} =="
if command -v ss >/dev/null 2>&1; then
  ss -ltn 2>/dev/null | awk -v port=":${CODEX_PROXY_PORT}" '$4 ~ port { print }'
elif command -v netstat >/dev/null 2>&1; then
  netstat -ltn 2>/dev/null | awk -v port=":${CODEX_PROXY_PORT}" '$4 ~ port { print }'
else
  echo "socket_tool=missing"
fi

if [ "${1:-}" = "--online" ]; then
  echo
  echo "== Online probes =="
  if command -v curl >/dev/null 2>&1; then
    for url in "https://api.openai.com" "https://github.com"; do
      code="$(curl -I -L -sS --max-time 10 -o /dev/null -w '%{http_code}' "${url}" 2>/dev/null || true)"
      echo "${url} http_code=${code:-failed}"
    done
  else
    echo "curl=missing"
  fi
fi
