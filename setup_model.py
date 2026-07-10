#!/usr/bin/env python3
"""Download all-MiniLM-L6-v2 ONNX model, tokenizer files, and Pico CSS."""

from __future__ import annotations

import sys
from pathlib import Path
import httpx

from pipeline import DEFAULT_ONNX_MODEL_DIR

MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
HF_BASE = f"https://huggingface.co/{MODEL_REPO}/resolve/main"
ONNX_BASE = f"https://huggingface.co/{MODEL_REPO}/resolve/main/onnx"

# Model files land in the shared cross-worktree location (see
# pipeline/config.py); Pico CSS stays per-checkout under templates/.
FILES = {
    f"{ONNX_BASE}/model.onnx": f"{DEFAULT_ONNX_MODEL_DIR}/model.onnx",
    f"{HF_BASE}/tokenizer.json": f"{DEFAULT_ONNX_MODEL_DIR}/tokenizer.json",
    f"{HF_BASE}/tokenizer_config.json": f"{DEFAULT_ONNX_MODEL_DIR}/tokenizer_config.json",
    f"{HF_BASE}/config.json": f"{DEFAULT_ONNX_MODEL_DIR}/config.json",
    f"{HF_BASE}/special_tokens_map.json": f"{DEFAULT_ONNX_MODEL_DIR}/special_tokens_map.json",
    f"{HF_BASE}/vocab.txt": f"{DEFAULT_ONNX_MODEL_DIR}/vocab.txt",
    "https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css": "templates/pico.min.css",
}


def download_model() -> None:
    # Check if all files exist and are non-empty
    all_exist = True
    for dest_path in FILES.values():
        dest = Path(dest_path)
        if not dest.exists() or dest.stat().st_size == 0:
            all_exist = False
            break

    if all_exist:
        print("Model, tokenizer files, and Pico CSS already exist.")
        return

    print("Downloading required files...")
    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        for url, dest_path in FILES.items():
            dest = Path(dest_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {dest.name} from {url}...")
            try:
                resp = client.get(url)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
            except Exception as e:
                print(f"Error downloading {dest.name}: {e}", file=sys.stderr)
                if dest.exists():
                    dest.unlink()
                raise e

    print("Successfully downloaded all files.")


if __name__ == "__main__":
    download_model()
