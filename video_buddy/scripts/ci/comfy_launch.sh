#!/usr/bin/env bash
# Launch Buddy's managed ComfyUI on :8188, check it, then stop it (also on failure).
# Run from video_buddy/ with VB_PY set to Buddy's venv python. ComfyUI flags come
# from COMFY_EXTRA_ARGS (the CI job sets --cpu --enable-manager).
#
#   scripts/ci/comfy_launch.sh <name> [--validate] [scan-log options...]
#
# The log and /system_stats are copied to ci-artifacts/ as comfyui-<name>.log
# and system_stats-<name>.json. --validate also checks the Klein stills graph.
set -euo pipefail

name=$1
shift
validate=0
if [ "${1:-}" = "--validate" ]; then
  validate=1
  shift
fi
check=("$VB_PY" scripts/ci/fresh_install_check.py)
log=state/logs/comfyui_8188.log
mkdir -p ci-artifacts

finish() {
  "$VB_PY" -m master_agent comfy stop >/dev/null 2>&1 || true
  cp "$log" "ci-artifacts/comfyui-$name.log" 2>/dev/null || true
}
trap finish EXIT

"$VB_PY" -m master_agent comfy start --no-watch
"${check[@]}" wait-ready --out "ci-artifacts/system_stats-$name.json"
# Manager fetches its channel in a background thread after the server is up.
sleep "${SETTLE_S:-20}"
"${check[@]}" scan-log "$log" "$@"
"${check[@]}" manager-live
if [ "$validate" = 1 ]; then
  "${check[@]}" validate-workflow --object-info-out "ci-artifacts/object_info-$name.json"
fi
