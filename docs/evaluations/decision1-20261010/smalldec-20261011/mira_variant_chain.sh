#!/bin/bash
set -u
while systemctl --user is-active -q julia-variants-20261011; do sleep 30; done
[ -f /tmp/smalldec/out/julia-variants/selection.json ] || { echo "mira variant FAILED: no selection"; exit 1; }
export HF_HUB_OFFLINE=1 PILOT_THREADS=4
cd /tmp
P=/tmp/smalldec/scripts/julia_variants.py
/tmp/laya-pilot/env/.venv/bin/python -I $P mira --out-dir /tmp/smalldec/out/julia-variants \
 && /tmp/laya-pilot/env/.venv/bin/python -I $P mira-report || echo "mira variant FAILED"
echo "mira variant done"
