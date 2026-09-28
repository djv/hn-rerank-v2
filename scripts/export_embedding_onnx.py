#!/usr/bin/env python3
"""Export a Hugging Face embedding model's encoder to ONNX for the bakeoff.

For models published without ONNX files (e.g. microsoft/harrier-oss-v1-*).
Writes ``onnx/model.onnx`` (+ ``model.onnx_data``) and the tokenizer to
--output-dir; pass that directory as ``--repo`` to
encode_replay_embeddings.py / bench_embed_gpu.py. The graph takes
input_ids + attention_mask and returns last_hidden_state; pooling stays in
the bakeoff code. Offline experiment tooling: torch is not a project
dependency, so run it in a throwaway environment, under ``batch``:

    batch uv run --no-project --index https://download.pytorch.org/whl/cpu \\
        --index-strategy unsafe-best-match --with torch --with transformers \\
        --with onnx python scripts/export_embedding_onnx.py --repo REPO
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path.home() / ".cache/hn-rerank-embedding-models"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    # torch and onnx are not project dependencies (see the docstring); via
    # importlib so type checking does not need them installed.
    onnx: Any = importlib.import_module("onnx")
    torch: Any = importlib.import_module("torch")
    from transformers import AutoModel, AutoTokenizer

    output = args.output_dir or DEFAULT_ROOT / (
        args.repo.replace("/", "--") + "--export"
    )
    (output / "onnx").mkdir(parents=True, exist_ok=True)
    tokenizer: Any = AutoTokenizer.from_pretrained(args.repo)
    tokenizer.save_pretrained(output)
    model: Any = AutoModel.from_pretrained(
        args.repo, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    class Encoder(torch.nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            return self.inner(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).last_hidden_state

    sample = tokenizer(["a short text", "a somewhat longer text"], padding=True)
    inputs = (
        torch.tensor(sample["input_ids"]),
        torch.tensor(sample["attention_mask"]),
    )
    raw = output / "onnx" / "raw.onnx"
    with torch.no_grad():
        torch.onnx.export(
            Encoder(model),
            inputs,
            str(raw),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "last_hidden_state": {0: "batch", 1: "sequence"},
            },
            opset_version=args.opset,
            dynamo=False,
        )
    # Re-save with every weight in one sibling file, whatever the exporter
    # chose (models over 2 GB cannot be a single protobuf).
    graph = onnx.load(str(raw), load_external_data=True)
    final = output / "onnx" / "model.onnx"
    onnx.save_model(
        graph,
        str(final),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx_data",
    )
    for leftover in (output / "onnx").iterdir():
        if leftover.name not in {"model.onnx", "model.onnx_data"}:
            leftover.unlink()
    print(f"wrote {final}")


if __name__ == "__main__":
    main()
