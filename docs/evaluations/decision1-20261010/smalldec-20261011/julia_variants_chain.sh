#!/bin/bash
set -u
while systemctl --user is-active -q finetune-fold0-20261011b; do sleep 30; done
echo "fine-tune finished; starting julia variants"
export HF_HUB_OFFLINE=1 PILOT_THREADS=4
PY="/tmp/smalldec/julia-venv/bin/python -I /tmp/smalldec/scripts/julia_variants.py"
cd /tmp && $PY select && $PY confirm && $PY report || echo "julia variants FAILED"
echo "julia variants done"
