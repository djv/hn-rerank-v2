#!/usr/bin/env python3
"""Give an ONNX export's RotaryEmbedding nodes explicit per-token positions.

Some onnx-community exports (e.g. EmbeddingGemma 2) feed
``com.microsoft::RotaryEmbedding`` a one-element ``position_ids`` of ``[0]``.
onnxruntime reads that as the start offset (token s gets position s), but
OpenVINO 2026.4 gives every token position 0, so its RoPE is the identity and
every layer after the first attention differs (cosine ~0.74 to onnxruntime on
CPU and GPU alike). This replaces that input with ``[batch, seq]`` positions
0..seq-1, which both runtimes read the same way.

Writes ``<model>.positions.onnx`` next to the model (graph only; it reuses the
weights file). Needs the ``onnx`` package, which is not a project dependency:
``uv run --group embedding-experiment --with onnx python scripts/patch_rotary_position_ids.py MODEL``.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="ONNX file, e.g. .../onnx/model.onnx")
    args = parser.parse_args()

    onnx: Any = importlib.import_module("onnx")
    numpy_helper: Any = importlib.import_module("onnx.numpy_helper")
    helper = onnx.helper
    proto = onnx.load(str(args.model), load_external_data=False)
    graph = proto.graph

    rotary = [n for n in graph.node if n.op_type == "RotaryEmbedding"]
    starts = {n.input[1] for n in rotary}
    # Only the position_ids constants: the weights stay in their external file.
    base_dir = str(args.model.parent)
    constants: dict[str, Any] = {
        t.name: numpy_helper.to_array(t, base_dir)
        for t in graph.initializer
        if t.name in starts
    }
    for node in graph.node:
        if node.op_type == "Constant" and node.output[0] in starts:
            value = helper.get_attribute_value(node.attribute[0])
            constants[node.output[0]] = numpy_helper.to_array(value, base_dir)
    for name in starts:
        value = constants.get(name)
        if value is None or value.size != 1 or int(value.reshape(-1)[0]) != 0:
            raise SystemExit(
                f"position_ids {name!r} is not a constant [0]; not patching"
            )
    input_ids = graph.input[0].name

    # positions[b, s] = s, shaped like input_ids.
    prefix = "/patch/positions"
    nodes = [
        helper.make_node("Shape", [input_ids], [f"{prefix}/shape"]),
        helper.make_node(
            "Gather", [f"{prefix}/shape", f"{prefix}/one"], [f"{prefix}/seq"], axis=0
        ),
        helper.make_node(
            "Range",
            [f"{prefix}/zero", f"{prefix}/seq", f"{prefix}/one"],
            [f"{prefix}/range"],
        ),
        helper.make_node(
            "Expand", [f"{prefix}/range", f"{prefix}/shape"], [f"{prefix}/ids"]
        ),
    ]
    graph.initializer.extend(
        [
            numpy_helper.from_array(np.array(0, dtype=np.int64), f"{prefix}/zero"),
            numpy_helper.from_array(np.array(1, dtype=np.int64), f"{prefix}/one"),
        ]
    )
    for node in rotary:
        node.input[1] = f"{prefix}/ids"
    # Prepend, so the graph stays topologically sorted.
    merged = nodes + list(graph.node)
    del graph.node[:]
    graph.node.extend(merged)

    output = args.model.with_name(args.model.stem + ".positions.onnx")
    onnx.save(proto, str(output))
    print(
        f"patched {len(rotary)} RotaryEmbedding nodes ({sorted(starts)}); wrote {output}"
    )


if __name__ == "__main__":
    main()
