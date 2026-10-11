#!/bin/bash
# Sequential Julia-1 / Mira-v1 pilots and gated 100/fold subset runs (run as one capped user unit).
set -u
J=/tmp/decision1-eval/work_heldout/paired_plans_v3/e37bdffbed7255febbada9b6e5b1a34479be700f6630b73a2cf8761e33ea5f6a/jobs.jsonl
S=/tmp/smalldec/scripts/smalldec_score.py
SUBSET=/tmp/laya-pilot/full/subset-100.json
OUT=/tmp/smalldec/out
GATE=2.5
export HF_HUB_OFFLINE=1 PILOT_THREADS=3

while systemctl --user is-active -q laya-subset100-20261011; do sleep 30; done
echo "laya run finished: $(systemctl --user show laya-subset100-20261011 -p Result --value)"

for b in julia mira; do
  if [ $b = julia ]; then PY=/tmp/smalldec/julia-venv/bin/python; M=/tmp/smalldec/models/Julia-1
  else PY=/tmp/laya-pilot/env/.venv/bin/python; M=/tmp/smalldec/models/Mira-v1; fi
  echo "== $b pilot"
  $PY -I $S $b $M $J $OUT/pilot-$b.json --pilot 8 || { echo "$b pilot FAILED"; continue; }
  rate=$($PY -I -c "import json; print(json.load(open('$OUT/pilot-$b.json'))['seconds_per_score'])")
  echo "$b pilot s/score $rate (gate $GATE)"
  if $PY -I -c "import sys; sys.exit(0 if $rate <= $GATE else 1)"; then
    echo "== $b subset run"
    $PY -I $S $b $M $J $OUT/scores-subset100-$b.jsonl --subset $SUBSET || { echo "$b run FAILED"; continue; }
    echo "== $b report"
    (cd ~/hn-rewrite/hn-rewrite-decision1 && .venv/bin/python /tmp/laya-pilot/scripts/laya_report.py \
      docs/evaluations/decision1-20261010/eval_decision1.py /tmp/decision1-eval/work_heldout \
      $OUT/scores-subset100-$b.jsonl $OUT/report-subset100-$b.json $SUBSET) || echo "$b report FAILED"
  else
    echo "$b slower than gate; no subset run"
  fi
done
echo "chain done"
