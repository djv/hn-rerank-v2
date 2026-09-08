#!/usr/bin/env python3
"""Download the production ONNX embedding model, tokenizer files, and Pico CSS.

Provisions DEFAULT_ONNX_MODEL_DIR with whatever model config.toml's
embedding_model_version currently names (mxbai-embed-xsmall-v1 as of
2026-08-27; see WORKLOG for the switch history). If a future model swap
changes MODEL_REPO here, also update DEFAULT_ONNX_MODEL_DIR/
DEFAULT_EMBEDDING_MODEL_VERSION/DEFAULT_EMBEDDING_MAX_TOKENS in
pipeline/config.py so the two stay in lockstep.
"""

from __future__ import annotations

import sys
from pathlib import Path
import httpx

from pipeline import DEFAULT_ONNX_MODEL_DIR

MODEL_REPO = "mixedbread-ai/mxbai-embed-xsmall-v1"
# Pinned HF revision the URLs below resolve at. Bump this (and re-run) for a
# deliberate model upgrade; embeddings stay valid across runs as long as the
# bytes behind the pin do not change (manifest-verified).
MODEL_REVISION = "main"
HF_BASE = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}"
ONNX_BASE = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/onnx"

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
    else:
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

    ensure_manifest()


def ensure_manifest() -> None:
    """Baseline (or verify) the model-directory manifest.

    First run grandfathers the current bytes; afterwards a mismatch means
    the files changed under a recorded baseline. This is an explicit
    operator action, so mismatch exits loud — the server itself
    warn-and-serves (see Embedder), but setup never silently re-baselines.
    """
    from pipeline.model_manifest import check_model_dir, manifest_path

    if manifest_path(DEFAULT_ONNX_MODEL_DIR).exists():
        _, mismatches, _ = check_model_dir(
            DEFAULT_ONNX_MODEL_DIR, MODEL_REPO, MODEL_REVISION
        )
        if mismatches:
            print(
                "Model directory differs from its manifest baseline:",
                file=sys.stderr,
            )
            for mismatch in mismatches:
                print(f"  - {mismatch}", file=sys.stderr)
            print(
                "Refusing to overwrite the baseline. If this change is "
                "deliberate, delete "
                f"{manifest_path(DEFAULT_ONNX_MODEL_DIR)} and re-run to "
                "re-baseline (stored vectors keep serving; see WORKLOG "
                "2026-09-07).",
                file=sys.stderr,
            )
            raise SystemExit(1)
        print("Model manifest verified.")
        return
    from pipeline.model_manifest import write_manifest

    manifest = write_manifest(DEFAULT_ONNX_MODEL_DIR, MODEL_REPO, MODEL_REVISION)
    print(
        f"Recorded model baseline manifest "
        f"({manifest.repo}@{manifest.revision}, "
        f"model.onnx sha256={manifest.files.get('model.onnx', '?')[:12]}...)."
    )


if __name__ == "__main__":
    download_model()
