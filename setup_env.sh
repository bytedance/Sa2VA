#!/usr/bin/env bash
# setup_env.sh — create a uv environment for a project, with the venv living in
# /tmp and symlinked back into the project (avoids a real .venv on slow mounts).
#
# Usage:
#   bash setup_env.sh [PROJECT] [EXTRA]
#     PROJECT  subdir under projects/ that holds a pyproject.toml   (default: sa2va)
#     EXTRA    uv optional-dependency group to sync                 (default: latest)
#
# Examples:
#   bash setup_env.sh                 # projects/sa2va, --extra=latest
#   bash setup_env.sh sa2va legacy    # projects/sa2va, --extra=legacy
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${1:-sa2va}"
EXTRA="${2:-latest}"

PROJECT_DIR="$REPO_ROOT/projects/$PROJECT"
if [ ! -f "$PROJECT_DIR/pyproject.toml" ]; then
  echo "[setup_env] no pyproject.toml under projects/$PROJECT" >&2
  echo "[setup_env] available projects:" >&2
  for d in "$REPO_ROOT"/projects/*/pyproject.toml; do
    [ -e "$d" ] && echo "    - $(basename "$(dirname "$d")")" >&2
  done
  exit 1
fi

# Load repo-root .env (HF_TOKEN, API keys, ...) if present.
if [ -f "$REPO_ROOT/.env" ]; then
  set -a; . "$REPO_ROOT/.env"; set +a
fi

# uv is installed once per machine by /opt/tiger/init; just make sure it is on PATH.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup_env] uv not found — it is normally installed by /opt/tiger/init." >&2
  echo "[setup_env] install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

# The venv lives in /tmp and is symlinked from the project directory.
VENV_TMP="/tmp/${PROJECT}_env"
VENV_LINK="$PROJECT_DIR/.venv"

# Create the /tmp venv if it is missing (e.g. a reboot wiped /tmp).
if [ ! -x "$VENV_TMP/bin/python" ]; then
  echo "[setup_env] creating venv at $VENV_TMP"
  ( cd "$PROJECT_DIR" && uv venv --python 3.11 "$VENV_TMP" )
fi

# (Re)point the project's .venv symlink at the /tmp venv.
if [ -e "$VENV_LINK" ] && [ ! -L "$VENV_LINK" ]; then
  echo "[setup_env] $VENV_LINK is a real directory, not a symlink — refusing to overwrite" >&2
  exit 1
fi
rm -f "$VENV_LINK"
ln -s "$VENV_TMP" "$VENV_LINK"

# Sync dependencies from the project directory.
echo "[setup_env] uv sync --extra=$EXTRA  (in projects/$PROJECT)"
( cd "$PROJECT_DIR" && uv sync --extra="$EXTRA" )

echo "[setup_env] done."
echo "[setup_env] activate with: source projects/$PROJECT/.venv/bin/activate"
