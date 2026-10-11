#!/bin/bash
# Wait for the Ettin fine-tune, then Julia tag pilot (50 stories) and, if under the gate, the full run.
set -u
while systemctl --user is-active -q finetune-tiny-20261011; do sleep 30; done
PY="/tmp/smalldec/julia-venv/bin/python -I /tmp/tagging/scripts/tagging.py"
cd /tmp
echo "julia tag pilot starting"
$PY julia --limit 50 | tee /tmp/tagging/julia-pilot.log || { echo "julia tag pilot FAILED"; exit 1; }
sps=$(grep -o '"s_per_story": [0-9.]*' /tmp/tagging/julia-pilot.log | tail -1 | awk '{print $2}')
echo "julia tag pilot s_per_story=$sps"
if awk -v s="$sps" 'BEGIN{exit !(s <= 2.1)}'; then
  echo "julia tag full run starting"
  $PY julia || { echo "julia tag full FAILED"; exit 1; }
  echo "julia tag full done"
else
  echo "julia tag GATE: over 2.1 s/story; full run not started"
fi
