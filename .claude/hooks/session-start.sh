#!/bin/bash
# SessionStart hook for Claude Code on the web: install deps and fetch the
# ONNX embedding model so the full test suite runs in cloud sessions.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}"

# Default group only (dev); embedding-experiment stays opt-in.
uv sync

# Tests read HN_ONNX_MODEL_DIR (pipeline/config.py); config.toml pins the
# server's onnx_model_dir to the host path, which gets a symlink below.
# Same files as setup_model.py minus Pico CSS (committed; jsdelivr may be blocked).
CONFIG_MODEL_DIR="/home/dev/hn-rewrite/shared/mxbai-embed-xsmall-v1"
MODEL_DIR="${HN_ONNX_MODEL_DIR:-$CONFIG_MODEL_DIR}"
HF_BASE="https://huggingface.co/mixedbread-ai/mxbai-embed-xsmall-v1/resolve/main"
mkdir -p "$MODEL_DIR"

fetch() {  # fetch <url> <dest>; skip if already present and non-empty
  local url="$1" dest="$2"
  [ -s "$dest" ] && return 0
  curl -fsSL --retry 3 -o "$dest.part" "$url" && mv "$dest.part" "$dest"
}

fetch_all() {
  fetch "$HF_BASE/onnx/model.onnx" "$MODEL_DIR/model.onnx" || return 1
  local f
  for f in tokenizer.json tokenizer_config.json config.json special_tokens_map.json vocab.txt; do
    fetch "$HF_BASE/$f" "$MODEL_DIR/$f" || return 1
  done
}

if ! fetch_all; then
  rm -f "$MODEL_DIR"/*.part
  # Non-fatal: without huggingface.co in the network allowlist, ~18
  # model-dependent tests error but everything else still works.
  echo "WARNING: model download failed (is huggingface.co allowed?)" >&2
fi

if [ -s "$MODEL_DIR/model.onnx" ] && [ ! -e "$CONFIG_MODEL_DIR" ]; then
  mkdir -p "$(dirname "$CONFIG_MODEL_DIR")"
  ln -s "$(realpath "$MODEL_DIR")" "$CONFIG_MODEL_DIR"
fi
