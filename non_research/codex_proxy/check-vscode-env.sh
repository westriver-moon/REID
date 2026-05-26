#!/usr/bin/env bash
set -u

print_env_for_pattern() {
  local label="$1"
  local pattern="$2"
  local pid
  pid="$(pgrep -u cgv841 -f "${pattern}" | head -n 1 || true)"
  echo "== ${label} =="
  echo "PID=${pid:-missing}"

  if [ -z "${pid:-}" ]; then
    return 0
  fi

  tr '\0' '\n' < "/proc/${pid}/environ" \
    | grep -E '^(ALL_PROXY|HTTPS_PROXY|HTTP_PROXY|NO_PROXY|TMPDIR|XDG_RUNTIME_DIR)=' \
    | sort || true
}

print_env_for_pattern "server-main" "out/server-main.js"
print_env_for_pattern "extensionHost" "type=extensionHost"
print_env_for_pattern "codex app-server" "codex app-server"
