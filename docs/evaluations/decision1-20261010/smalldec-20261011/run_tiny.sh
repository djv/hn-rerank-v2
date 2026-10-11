#!/bin/bash
while systemctl --user is-active -q mira-variant-20261011; do sleep 15; done
echo "starting ettin-17m 5-fold fine-tune"
export HF_HUB_OFFLINE=1 PILOT_THREADS=4
exec /tmp/laya-pilot/env/.venv/bin/python -I /tmp/finetune/ft_tiny.py --epochs 2 --max-tokens 160 \
  --model-dir /tmp/laya-pilot/hf/hub/models--jhu-clsp--ettin-encoder-17m/snapshots/987607455c61e7a5bbc85f7758e0512ea6d0ae4c \
  --out /tmp/finetune/tiny-ettin17m.json
