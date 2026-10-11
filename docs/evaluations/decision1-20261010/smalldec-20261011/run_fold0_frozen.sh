#!/bin/bash
while systemctl --user is-active -q julia-variants-20261011 || systemctl --user is-active -q mira-variant-20261011; do sleep 30; done
echo "variants finished; starting fold-0 fine-tune (frozen embeddings)"
export HF_HOME=/tmp/laya-pilot/hf HF_HUB_OFFLINE=1 PILOT_THREADS=4
exec /tmp/laya-pilot/env/.venv/bin/python -I /tmp/finetune/ft_fold.py --db /tmp/finetune/snapshot.db --revision abc32620dd4f6ab06f5fbe905dc25f310618e09f --out /tmp/finetune/fold0.json
