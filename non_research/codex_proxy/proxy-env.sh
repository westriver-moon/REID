#!/usr/bin/env sh
# Single source of proxy environment for VS Code Server and shell sessions.

CODEX_PROXY_HOST="${CODEX_PROXY_HOST:-127.0.0.1}"
CODEX_PROXY_PORT="${CODEX_PROXY_PORT:-7897}"
CODEX_PROXY_SCHEME="${CODEX_PROXY_SCHEME:-socks5h}"
CODEX_PROXY_TIMEOUT="${CODEX_PROXY_TIMEOUT:-2}"
CODEX_PROXY_TMPDIR="${CODEX_PROXY_TMPDIR:-/home/cgv841/ybj/codex-local/tmp}"
CODEX_PROXY_NO_PROXY="${CODEX_PROXY_NO_PROXY:-localhost,127.0.0.1,::1}"
CODEX_IPC_DIR="${CODEX_IPC_DIR:-/home/cgv841/ybj/codex-local/tmp/codex-ipc}"

codex_proxy_url() {
  printf '%s://%s:%s\n' "${CODEX_PROXY_SCHEME}" "${CODEX_PROXY_HOST}" "${CODEX_PROXY_PORT}"
}

codex_proxy_port_open() {
  timeout "${CODEX_PROXY_TIMEOUT}" bash -c \
    'host="$1"; port="$2"; : < "/dev/tcp/${host}/${port}"' \
    _ "${CODEX_PROXY_HOST}" "${CODEX_PROXY_PORT}" >/dev/null 2>&1
}

codex_proxy_apply() {
  mkdir -p "${CODEX_PROXY_TMPDIR}"
  export TMPDIR="${CODEX_PROXY_TMPDIR}"
  export XDG_RUNTIME_DIR="${CODEX_PROXY_TMPDIR}"
  mkdir -p "${CODEX_IPC_DIR}"
  chmod 700 "${CODEX_IPC_DIR}" 2>/dev/null || true

  proxy_url="$(codex_proxy_url)"

  if codex_proxy_port_open; then
    export ALL_PROXY="${proxy_url}"
    export HTTPS_PROXY="${proxy_url}"
    export HTTP_PROXY="${proxy_url}"
    export all_proxy="${proxy_url}"
    export https_proxy="${proxy_url}"
    export http_proxy="${proxy_url}"
    export NO_PROXY="${CODEX_PROXY_NO_PROXY}"
    export no_proxy="${CODEX_PROXY_NO_PROXY}"
    return 0
  fi

  unset ALL_PROXY HTTPS_PROXY HTTP_PROXY all_proxy https_proxy http_proxy NO_PROXY no_proxy
  return 1
}

codex_proxy_status() {
  printf 'CODEX_PROXY=%s\n' "$(codex_proxy_url)"
  printf 'CODEX_PROXY_PORT_OPEN='
  if codex_proxy_port_open; then
    printf 'yes\n'
  else
    printf 'no\n'
  fi
  printf 'TMPDIR=%s\n' "${TMPDIR:-}"
  printf 'CODEX_IPC_DIR=%s\n' "${CODEX_IPC_DIR}"
  printf 'ALL_PROXY=%s\n' "${ALL_PROXY:-}"
  printf 'HTTPS_PROXY=%s\n' "${HTTPS_PROXY:-}"
  printf 'HTTP_PROXY=%s\n' "${HTTP_PROXY:-}"
  printf 'NO_PROXY=%s\n' "${NO_PROXY:-}"
}

if [ "$(basename "$0")" = "proxy-env.sh" ]; then
  codex_proxy_apply >/dev/null 2>&1 || true
  codex_proxy_status
fi
