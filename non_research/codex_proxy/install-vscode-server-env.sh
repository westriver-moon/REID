#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="/home/cgv841/.vscode-server/server-env-setup"
BACKUP_DIR="${SCRIPT_DIR}/backups"
STAMP="$(date +%Y%m%d-%H%M%S)"

mkdir -p "${BACKUP_DIR}"

if [ -e "${TARGET}" ]; then
  cp -p "${TARGET}" "${BACKUP_DIR}/server-env-setup.${STAMP}.bak"
fi

tmp="${TARGET}.tmp.${STAMP}"
cat > "${tmp}" <<'ENVEOF'
#!/usr/bin/env bash
# Managed by /home/cgv841/ybj/non_research/codex_proxy/install-vscode-server-env.sh

if [ -r /home/cgv841/ybj/non_research/codex_proxy/proxy-env.sh ]; then
  # shellcheck source=/home/cgv841/ybj/non_research/codex_proxy/proxy-env.sh
  . /home/cgv841/ybj/non_research/codex_proxy/proxy-env.sh
  codex_proxy_apply >/dev/null 2>&1 || true
fi
ENVEOF

chmod 700 "${tmp}"
mv "${tmp}" "${TARGET}"
echo "installed ${TARGET}"
echo "backup_dir=${BACKUP_DIR}"
