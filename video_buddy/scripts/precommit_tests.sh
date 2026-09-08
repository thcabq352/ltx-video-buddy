#!/usr/bin/env bash
# Runs the VIDEO BUDDY test suite before every commit.
# Exit non-zero (blocking the commit) on any test failure.
#
# Versioned script that a local-git hook (scripts/../.git/hooks/pre-commit)
# invokes, so the gate is reproducible for anyone who clones the repo.
#
# Skip with:   SKIP_TESTS=1 git commit    (or)   git commit --no-verify

set -euo pipefail

if [[ "${SKIP_TESTS:-0}" == "1" ]]; then
  echo "[pre-commit] SKIP_TESTS=1 — skipping test suite"
  exit 0
fi

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${here}/.." && pwd)"
py="${project_root}/.venv/Scripts/python.exe"

cd "${project_root}"
if [[ ! -f "${py}" ]]; then
  echo "[pre-commit] ERROR: venv python not found at ${py}"
  echo "[pre-commit] create it with: python -m venv .venv && .venv/Scripts/python.exe -m pip install -r requirements.txt"
  exit 1
fi

echo "[pre-commit] running test suite (venv=${py}) ..."
if ! "${py}" -m pytest tests/ -q; then
  echo "[pre-commit] FAILED — tests are red; fix before committing."
  exit 1
fi
echo "[pre-commit] tests green."
