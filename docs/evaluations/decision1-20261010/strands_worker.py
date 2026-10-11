#!/usr/bin/env python3
"""Strands Decider batch scorer (offline, CPU).

Reads content-bound jobs with cache_identity/request_sha256 metadata. Only
validated cache entries are skipped; legacy, stale or malformed entries fail.
New metadata-bearing records are published atomically without overwriting.
Jobs that share a rendered-state text
prefix (the HN fold taste profile, the dating taste text) reuse one encoded
prefix cache: the prefix is forwarded once per group, then each job's remaining
state + question is forwarded against a forked copy of that cache.  This is the
package's own shared-prefix mechanism (infer._expand_cache), applied across
states instead of across questions; `--check N` compares it with the package's
plain path (engine.evaluate) on N jobs.

No downloads: both adapter and base must already exist at immutable local
snapshot revisions. The base loader is explicitly pinned to its local snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version as installed_version
from pathlib import Path
from unittest.mock import patch

# Must precede third-party imports; every Hub resolution below is local-only too.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from score_cache import (
    CacheIdentity,
    CacheValidationError,
    read_cached_score,
    request_sha256,
    write_cached_score,
)

import torch

from strands_decider.infer import _expand_cache, load_engine
from strands_decider.modeling import (
    StrandsDeciderModel,
    apply_temperature,
    gather_options,
    masked_log_softmax,
    pool_last_token,
)
from strands_decider.prompting import read_noul, render_question, render_state
from strands_decider.schema import NoulQuestion, SystemOneRequest

MODEL_ID = "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
MODEL_REVISION = "216d1ccb46a9e9b036eb9187b248db98480f77d4"
BASE_MODEL_ID = "Qwen/Qwen3.5-2B-Base"
BASE_MODEL_REVISION = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
# Exact preserved probe environment. These constants are part of the scorer's
# source digest; drift must fail before producing a score under that identity.
EXPECTED_RUNTIME_PACKAGES = {
    "accelerate": "1.15.0",
    "annotated-doc": "0.0.5",
    "annotated-types": "0.8.0",
    "anyio": "4.15.1",
    "click": "8.5.0",
    "fastapi": "0.143.0",
    "filelock": "3.32.3",
    "fsspec": "2026.7.0",
    "h11": "0.16.0",
    "hf-xet": "1.7.0",
    "httpcore2": "2.13.1",
    "httptools": "0.9.0",
    "httpx2": "2.13.1",
    "huggingface-hub": "2.2.0",
    "idna": "3.20",
    "jinja2": "3.1.6",
    "markdown-it-py": "4.2.0",
    "markupsafe": "3.0.3",
    "mdurl": "0.1.2",
    "mpmath": "1.3.0",
    "networkx": "3.6.1",
    "numpy": "2.5.4",
    "opentelemetry-api": "1.45.1",
    "packaging": "26.3",
    "peft": "0.21.2",
    "psutil": "7.2.2",
    "pydantic": "2.14.0",
    "pydantic-core": "2.50.0",
    "pygments": "2.21.0",
    "python-dotenv": "1.2.4",
    "pyyaml": "6.0.3",
    "regex": "2026.9.29",
    "rich": "15.0.0",
    "safetensors": "0.8.0",
    "setuptools": "78.1.0",
    "shellingham": "1.5.4",
    "starlette": "1.7.0",
    "strands-decider": "0.1.0",
    "sympy": "1.14.0",
    "tokenizers": "0.23.3",
    "torch": "2.14.1+cpu",
    "tqdm": "4.70.1",
    "transformers": "5.19.0",
    "truststore": "0.10.4",
    "typer": "0.27.3",
    "typing-extensions": "4.16.0",
    "typing-inspection": "0.4.4",
    "uvicorn": "0.54.0",
    "uvloop": "0.23.0",
    "watchfiles": "1.3.0",
    "websockets": "17.2",
}


def validate_runtime_packages() -> dict[str, str]:
    observed = {}
    for name, expected in EXPECTED_RUNTIME_PACKAGES.items():
        try:
            actual = installed_version(name)
        except PackageNotFoundError as exc:
            raise CacheValidationError(
                f"required pinned inference package is absent: {name}"
            ) from exc
        if actual != expected:
            raise CacheValidationError(
                f"inference package {name} is {actual}; expected pinned {expected}"
            )
        observed[name] = actual
    return observed


def worker_identity(model_id: str, revision: str, dtype: str) -> CacheIdentity:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise CacheValidationError(
            "model revision must be an immutable lowercase Hub commit"
        )
    return CacheIdentity(
        backend="strands",
        model_id=model_id,
        model_revision=revision,
        dtype=dtype,
        scorer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )


def validated_pending_jobs(jobs: list[dict], identity: CacheIdentity) -> list[dict]:
    """Validate every request and existing entry before any model is loaded."""
    pending = []
    for index, job in enumerate(jobs):
        try:
            if job.get("cache_identity") != asdict(identity):
                raise CacheValidationError(
                    "job cache identity differs from worker identity"
                )
            digest = request_sha256(job["state"], job["question"], identity)
            if job.get("request_sha256") != digest:
                raise CacheValidationError("job request digest is missing or stale")
            path = Path(job["out"])
            if path.name != f"{digest}.json":
                raise CacheValidationError(
                    "cache filename must equal the request digest"
                )
            if path.exists() or path.is_symlink():
                read_cached_score(path, job["state"], job["question"], identity)
            else:
                pending.append(job)
        except (KeyError, TypeError, AttributeError) as exc:
            raise CacheValidationError(
                f"malformed scoring job at index {index}"
            ) from exc
    return pending


def local_snapshot(model_id: str, revision: str) -> Path:
    """Resolve an immutable snapshot already on disk; never acquire weights."""
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise CacheValidationError("snapshot revision must be an immutable Hub commit")
    from huggingface_hub import snapshot_download

    try:
        path = Path(
            snapshot_download(
                repo_id=model_id, revision=revision, local_files_only=True
            )
        ).resolve()
    except (OSError, ValueError) as exc:
        raise CacheValidationError(
            f"pinned local snapshot unavailable for {model_id}@{revision}; no download allowed"
        ) from exc
    if not path.is_dir() or path.name != revision or path.parent.name != "snapshots":
        raise CacheValidationError(
            "Hub did not resolve the requested immutable local snapshot"
        )
    return path


def common_prefix(texts: list[str]) -> str:
    p = os.path.commonprefix(texts)
    return p[: p.rfind("\n") + 1]


def fork_prefix_cache(cache, n: int):
    """Fork without aliasing the singleton recurrent prefix state.

    Package 0.1.0 expands then calls contiguous(); for n=1 that returns
    the original tensor. Transformers updates conv/recurrent states in place.
    """
    fork = _expand_cache(cache, n)
    if n == 1:
        for layer in fork.layers:
            for name in ("keys", "values", "conv_states", "recurrent_states"):
                value = getattr(layer, name, None)
                if isinstance(value, torch.Tensor):
                    setattr(layer, name, value.clone())
                elif isinstance(value, dict):
                    setattr(
                        layer,
                        name,
                        {
                            k: v.clone() if isinstance(v, torch.Tensor) else v
                            for k, v in value.items()
                        },
                    )
    return fork


class Scorer:
    def __init__(self, model_id: str, dtype: str, *, base_snapshot: Path) -> None:
        from strands_decider import infer

        if dtype in ("bfloat16", "int8", "bf16w-fp32"):
            # Keep bf16 weights: the package upcasts the torso to fp32 on CPU, which
            # needs ~7.5 GB and does not fit this host.
            infer.SystemOneEngine._upcast_torso_for_cpu = lambda self: None
        original_load_torso = StrandsDeciderModel._load_torso
        pinned_base_used = False

        def load_pinned_torso(config, device_map, attn_implementation):
            nonlocal pinned_base_used
            if config.base_model != BASE_MODEL_ID:
                raise CacheValidationError(
                    "adapter base model differs from the pinned base"
                )
            config.base_model = str(base_snapshot)
            if hasattr(config, "base_revision"):
                config.base_revision = None
            pinned_base_used = True
            return original_load_torso(config, device_map, attn_implementation)

        with patch.object(
            StrandsDeciderModel, "_load_torso", staticmethod(load_pinned_torso)
        ):
            self.engine = load_engine(model_id, device="cpu")
        if not pinned_base_used or self.engine.model.config.base_model != str(
            base_snapshot
        ):
            raise CacheValidationError(
                "engine did not use the pinned local base snapshot"
            )
        self.tok = self.engine.tok
        self.model = self.engine.model
        if dtype == "int8":
            self._quantize_int8()
        if dtype == "bf16w-fp32":
            self._fp32_compute()

    @torch.inference_mode()
    def _fp32_compute(self) -> None:
        """bf16-stored Linear weights, fp32 compute: the package's own CPU numerics
        (it upcasts the whole torso to fp32) without the ~7.5 GB fp32 copy, which
        does not fit this host.  Each Linear upcasts its weight per call.  bf16
        matmuls on this AVX2-only CPU run ~5x slower than fp32."""
        import torch.nn as nn
        import torch.nn.functional as F

        def forward(self, x):
            w = self.weight if self.weight.dtype == x.dtype else self.weight.to(x.dtype)
            b = (
                self.bias
                if self.bias is None or self.bias.dtype == x.dtype
                else self.bias.to(x.dtype)
            )
            return F.linear(x, w, b)

        nn.Linear.forward = forward
        torso = self.model.torso
        emb = torso.get_input_embeddings()
        for mod in torso.modules():
            if mod is emb or isinstance(mod, nn.Linear):
                continue
            for pname, prm in list(mod.named_parameters(recurse=False)):
                if prm.dtype != torch.float32:
                    setattr(mod, pname, nn.Parameter(prm.float(), requires_grad=False))
            for bname, buf in list(mod.named_buffers(recurse=False)):
                if (
                    buf is not None
                    and buf.is_floating_point()
                    and buf.dtype != torch.float32
                ):
                    setattr(mod, bname, buf.float())
        emb.register_forward_hook(lambda m, i, o: o.float())

    @torch.inference_mode()
    def _quantize_int8(self) -> None:
        """Merge the LoRA adapter, then int8 dynamic quantisation (per output channel) of every torso Linear.

        bf16 matmuls on this AVX2-only CPU run at ~23 GFLOPS (5x slower than fp32, 11x
        slower than int8 dynamic), and an fp32 torso (~7.5 GB) does not fit in RAM.
        Each Linear is upcast to fp32 one at a time and replaced by its int8 dynamic
        version; the remaining small tensors run in fp32; the embedding table stays
        bf16 and its output is cast to fp32.  An approximation of the published
        model: compare with --dtype bfloat16 on a sample.
        """
        import torch.nn as nn
        from torch.ao.nn.quantized.dynamic import Linear as QLinear
        from torch.ao.quantization import per_channel_dynamic_qconfig

        def rss() -> str:
            return (
                open("/proc/self/status")
                .read()
                .split("VmRSS:")[1]
                .split("\n")[0]
                .strip()
            )

        print(f"int8: before merge rss {rss()}", flush=True)
        torso = self.model.torso.merge_and_unload()
        print(f"int8: merged rss {rss()}", flush=True)
        n = 0
        # Names, not module objects: holding the original Linears would keep every
        # fp32 upcast alive.
        names = [k for k, m in torso.named_modules() if isinstance(m, nn.Linear)]
        for full in names:
            parent_name, _, name = full.rpartition(".")
            parent = torso.get_submodule(parent_name) if parent_name else torso
            child = getattr(parent, name)
            child.float()
            child.qconfig = per_channel_dynamic_qconfig
            setattr(parent, name, QLinear.from_float(child))
            del child
            n += 1
            if n % 80 == 0:
                print(f"int8: {n} layers rss {rss()}", flush=True)
        emb = torso.get_input_embeddings()
        for mod in torso.modules():
            if mod is emb:
                continue
            for pname, prm in list(mod.named_parameters(recurse=False)):
                if prm.dtype != torch.float32:
                    setattr(mod, pname, nn.Parameter(prm.float(), requires_grad=False))
            for bname, buf in list(mod.named_buffers(recurse=False)):
                if (
                    buf is not None
                    and buf.is_floating_point()
                    and buf.dtype != torch.float32
                ):
                    setattr(mod, bname, buf.float())
        import ctypes
        import gc

        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        print(f"int8: cast rss {rss()}", flush=True)
        emb.register_forward_hook(lambda m, i, o: o.float())
        torso.config.dtype = torch.float32
        self.model.torso = torso
        print(f"int8: quantised {n} Linear layers", flush=True)

    @torch.inference_mode()
    def plain(self, state: object, question: dict) -> float:
        name = next(iter(question))
        resp = self.engine.evaluate(
            SystemOneRequest(
                state=state, questions={name: NoulQuestion(**question[name])}
            )
        )
        return float(resp.answers[name].noul)

    @torch.inference_mode()
    def prefix(self, text: str):
        ids = self.tok(text, add_special_tokens=True)["input_ids"]
        t = torch.tensor([ids])
        out = self.model.torso(
            input_ids=t,
            attention_mask=torch.ones_like(t),
            use_cache=True,
            return_dict=True,
        )
        return ids, out.past_key_values

    @torch.inference_mode()
    def batch(
        self, prefix_ids: list[int], cache, items: list[tuple[str, dict]]
    ) -> list[tuple[float, int]]:
        """items: (state_text, question dict). All state texts start with the prefix text."""
        eng = self.engine
        P = len(prefix_ids)
        rows, opt_rows, rendered, kinds = [], [], [], []
        for state_text, question in items:
            name = next(iter(question))
            rq = render_question(NoulQuestion(**question[name]))
            s, q = eng._fit(state_text, [rq.text])
            if s[:P] != prefix_ids:
                raise ValueError("prefix tokens differ; use the plain path")
            rows.append(s[P:] + q[0])
            opt_rows.append(eng._option_idx([rq], len(s) - P)[0].tolist())
            rendered.append(rq)
            kinds.append(rq.kind)
        n = len(rows)
        fork = fork_prefix_cache(cache, n)
        ids, mask = eng._pad(rows)
        full_mask = torch.cat([torch.ones(n, P, dtype=mask.dtype), mask], dim=1)
        hidden = self.model.encode(
            input_ids=ids, attention_mask=full_mask, past_key_values=fork
        )
        pooled = pool_last_token(hidden, full_mask).to(torch.float32)
        width = max(len(r) for r in opt_rows)
        opt_idx = torch.tensor(
            [r + [-1] * (width - len(r)) for r in opt_rows], dtype=torch.long
        )
        options = gather_options(hidden, opt_idx)
        raw = self.model.head(pooled, options.to(torch.float32))
        logits = apply_temperature(raw, eng._temperatures(kinds))
        probs = masked_log_softmax(
            logits, torch.tensor([r.n_slots for r in rendered])
        ).exp()
        return [
            (
                read_noul(probs[i, : rendered[i].n_slots].tolist(), rendered[i]),
                P + len(rows[i]),
            )
            for i in range(n)
        ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", type=Path)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--model-revision", default=MODEL_REVISION)
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float32", "int8", "bf16w-fp32"),
    )
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument(
        "--check",
        type=int,
        default=0,
        help="compare N cached results with the plain path",
    )
    ap.add_argument(
        "--group-key", default="group", help="job field naming its shared-prefix group"
    )
    ap.add_argument(
        "--bucket-by-length",
        action="store_true",
        help="sort within each profile group by fitted prompt length to reduce padding",
    )
    a = ap.parse_args()
    if a.batch < 1 or a.threads < 1 or (a.limit is not None and a.limit < 0):
        raise CacheValidationError(
            "batch/threads must be positive and limit nonnegative"
        )
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    jobs = [
        json.loads(line) for line in a.jobs.read_text().splitlines() if line.strip()
    ]
    identity = worker_identity(a.model, a.model_revision, a.dtype)
    todo = validated_pending_jobs(jobs, identity)[: a.limit]
    print(f"{len(todo)} uncached of {len(jobs)} jobs", flush=True)
    if not todo:
        return
    runtime_packages = validate_runtime_packages()
    model_snapshot = local_snapshot(a.model, a.model_revision)
    base_snapshot = local_snapshot(BASE_MODEL_ID, BASE_MODEL_REVISION)
    t0 = time.time()
    sc = Scorer(str(model_snapshot), a.dtype, base_snapshot=base_snapshot)
    print(
        f"loaded {a.model} ({a.dtype}) in {time.time() - t0:.0f}s, threads {a.threads}",
        flush=True,
    )
    groups: dict[str, list[dict]] = {}
    for j in todo:
        groups.setdefault(str(j.get(a.group_key, "")), []).append(j)
    done = 0
    t_start = time.time()
    for key, group in groups.items():
        texts = [render_state(j["state"]) for j in group]
        if a.bucket_by_length:

            def fitted_length(pair):
                j, text = pair
                name = next(iter(j["question"]))
                rq = render_question(NoulQuestion(**j["question"][name]))
                state_ids, question_ids = sc.engine._fit(text, [rq.text])
                return len(state_ids) + len(question_ids[0])

            ordered = sorted(zip(group, texts, strict=True), key=fitted_length)
            group = [j for j, _ in ordered]
            texts = [text for _, text in ordered]
        pref = common_prefix(texts) if len(group) > 1 else ""
        t1 = time.time()
        prefix_ids, cache = sc.prefix(pref) if pref else ([], None)
        print(
            f"group {key!r}: {len(group)} jobs, prefix {len(prefix_ids)} tokens in {time.time() - t1:.1f}s",
            flush=True,
        )
        for start in range(0, len(group), a.batch):
            chunk = group[start : start + a.batch]
            tb = time.time()
            try:
                if cache is None:
                    raise ValueError("no shared prefix")
                res = sc.batch(
                    prefix_ids,
                    cache,
                    [(t, j["question"]) for t, j in zip(texts[start:], chunk)],
                )
                cached = True
            except ValueError as e:
                print(f"  plain path for {len(chunk)} jobs: {e}", flush=True)
                res = [(sc.plain(j["state"], j["question"]), -1) for j in chunk]
                cached = False
            for j, (p, ntok) in zip(chunk, res, strict=True):
                write_cached_score(
                    Path(j["out"]),
                    p,
                    j["state"],
                    j["question"],
                    identity,
                    extra_fields={
                        "n_tokens": ntok,
                        "prefix_cached": cached,
                        "base_model_id": BASE_MODEL_ID,
                        "base_model_revision": BASE_MODEL_REVISION,
                        "runtime_packages": runtime_packages,
                    },
                )
            done += len(chunk)
            el = time.time() - t_start
            print(
                f"  {done}/{len(todo)} batch {time.time() - tb:.1f}s; {el / done:.2f}s/job; eta {el / done * (len(todo) - done) / 60:.0f} min",
                flush=True,
            )
        if a.check:
            for j, t in list(zip(group, texts))[: a.check]:
                tp = time.time()
                p_plain = sc.plain(j["state"], j["question"])
                p_cached = read_cached_score(
                    Path(j["out"]), j["state"], j["question"], identity
                )
                print(
                    f"  check: cached {p_cached:.4f} plain {p_plain:.4f} diff {p_cached - p_plain:+.4f} ({time.time() - tp:.1f}s plain)",
                    flush=True,
                )
            a.check = 0
    print(f"done {done} jobs in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
