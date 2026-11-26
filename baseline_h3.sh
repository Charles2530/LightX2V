#!/usr/bin/env bash
# H3-20: four distilled steps + Sage2 SLA 90% + vision patch Linear + LightVAE.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${H3_PYTHON:-$ROOT/.venv_h3_runtime/bin/python}" "$ROOT/scripts/minimax_h3/baseline_h3.py" "$@"
