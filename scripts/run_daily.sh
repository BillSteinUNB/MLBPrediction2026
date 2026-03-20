#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_EXE="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON_EXE}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/Scripts/python.exe" ]]; then
    PYTHON_EXE="${REPO_ROOT}/.venv/Scripts/python.exe"
  else
    echo "Project virtual environment not found at ${PYTHON_EXE}." >&2
    exit 1
  fi
fi

cd "${REPO_ROOT}"
exec "${PYTHON_EXE}" -m src.pipeline.daily --date today --mode prod "$@"
