#!/usr/bin/env bash
# VIDEO BUDDY — macOS / Linux installer
set -euo pipefail
cd "$(dirname "$0")"
if command -v python3 >/dev/null 2>&1; then
  exec python3 install.py "$@"
fi
if command -v python >/dev/null 2>&1; then
  exec python install.py "$@"
fi
echo "FAIL  Python 3.10+ not found."
echo "  macOS:  brew install python"
echo "  Debian: sudo apt install python3 python3-venv python3-pip"
echo "  Fedora: sudo dnf install python3"
exit 1
