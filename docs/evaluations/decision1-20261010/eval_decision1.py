#!/usr/bin/env python3
"""Offline test of Microsoft-Decision-1 (OpenRouter Decisions API) as an HN reranker.

Phases (all files live in one work directory):

  folds     Run the canonical evaluator (scripts/eval_ranker_variants.py) on the
            production variant, capturing each fold's held-out candidates,
            labels, production scores and a taste profile built ONLY from that
            fold's training feedback.  Free; no network.
  estimate  Project token use and cost for scoring every candidate.
  score     Ask Decision-1 "Would this user upvote this story?" for each
            (fold, story); responses are cached on disk, spend is capped.
  prepare   Write complete real/empty Strands job manifests without inference.
  audit     Validate exact paired cache coverage without quality metrics.
  report    Strands paired metrics and fixed blends only after full validation.
            Legacy Decision-1 efficacy reports are disabled pending provenance.

Backends (--backend): ``decision1`` (default, OpenRouter, paid) or ``strands``
(Strands Decider 2B, free open weights, run locally on CPU by
strands_worker.py inside a separate venv given by --strands-python).  Both use
the same state, question, folds and metric code; strands responses go to
versioned content-bound ``<work>/cache_strands_v2/{real,empty}`` entries.
Prepare paired jobs offline with ``prepare``; ``audit`` rejects partial/stale
coverage. Reports require both profiles and a frozen --snapshot, and use one
fixed blend weight. Legacy caches are preserved but cannot establish provenance.

Not wired into serving.  Story points and comment counts are never sent.
"""

from __future__ import annotations

import argparse
import subprocess
import json
import math
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlparse

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_cache import (  # noqa: E402
    CacheIdentity,
    CacheValidationError,
    cache_path,
    read_cached_score,
    request_sha256,
)
from frozen_inputs import (  # noqa: E402
    FrozenInputs,
    FrozenInputsError,
    RECEIPT_FILENAME,
    freeze_inputs,
    hash_file,
    publish_immutable,
    publish_result,
    read_frozen_inputs,
    validate_frozen_inputs,
    verify_immutable,
)

from database import Database, Story  # noqa: E402
from pipeline import Config, NULL_TRACE, RankTrace, _NullTrace  # noqa: E402
from scripts import eval_ranker_variants as evaluator  # noqa: E402

MODEL = "microsoft/microsoft-decision-1"
STRANDS_MODEL = "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
STRANDS_WORKER = Path(__file__).resolve().parent / "strands_worker.py"
STRANDS_REVISION = "216d1ccb46a9e9b036eb9187b248db98480f77d4"
STRANDS_DTYPE = "bf16w-fp32"
PRIMARY_BLEND_WEIGHT = 0.2
BACKEND = "decision1"  # set from --backend in main()
ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
CREDITS_URL = "https://openrouter.ai/api/v1/credits"
PRICE_PER_INPUT_TOKEN = 0.042e-6
ENV_PATH = Path("/home/dev/hn-rewrite/shared/.env")
UP, NEUTRAL, DOWN = 2, 1, 0
PROFILE_UP = 60
PROFILE_DOWN = 40
TEXT_CHARS = 500
QUESTION = {
    "upvote": {
        "type": "noul",
        "instructions": (
            "Would this Hacker News reader upvote this story? Judge only from the "
            "reader's past upvoted and downvoted stories in reader_profile."
        ),
        "criteria": {
            "true": "The story fits the reader's demonstrated interests; they would upvote it.",
            "false": "The reader would skip it or downvote it.",
        },
    }
}


@dataclass(frozen=True)
class Card:
    id: int
    title: str
    domain: str
    source: str
    text: str


@dataclass(frozen=True)
class Fold:
    fold: int
    candidates: list[Card]
    labels: list[int]  # -1 unjudged, 0 down, 1 neutral, 2 up
    production: list[float]
    profile_up: list[str]
    profile_down: list[str]
    profile_ids: list[int]
    train_counts: dict[str, int]


def _domain(url: str | None) -> str:
    if not url:
        return ""
    host = urlparse(url).netloc.lower()
    return host.removeprefix("www.")


def _card(story: Story) -> Card:
    body = story.self_text or story.article_body or story.text_content or ""
    text = " ".join(body.split())[:TEXT_CHARS]
    return Card(story.id, story.title, _domain(story.url), story.source, text)


def _profile_line(story: Story) -> str:
    domain = _domain(story.url)
    return f"{story.title} ({domain})" if domain else story.title


def _build_profile(
    fold: evaluator.FoldData,
) -> tuple[list[str], list[str], list[int]]:
    """Most recent up/down votes from this fold's training rows only."""
    order = np.argsort(-np.asarray(fold.train_vote_times, dtype=float), kind="stable")
    candidate_ids = {s.id for s in fold.candidates}
    up: list[str] = []
    down: list[str] = []
    ids: list[int] = []
    for i in order:
        story = fold.train_stories[i]
        label = int(fold.y_train[i])
        if story.id in candidate_ids:
            raise ValueError(f"training story {story.id} is also a candidate")
        if label == UP and len(up) < PROFILE_UP:
            up.append(_profile_line(story))
            ids.append(story.id)
        elif label == DOWN and len(down) < PROFILE_DOWN:
            down.append(_profile_line(story))
            ids.append(story.id)
    return up, down, ids


def cmd_folds(work: Path, evaluator_args: Sequence[str]) -> None:
    work.mkdir(parents=True, exist_ok=True)
    if (work / "folds.json").exists() or (work / RECEIPT_FILENAME).exists():
        raise ValueError("folds already exist; use a new work directory")
    captured: list[Fold] = []
    original = evaluator._production_scores

    def score(
        fold: evaluator.FoldData, config: Config, source_db: Database | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        result = original(fold, config, source_db)
        judged = {
            s.id: int(a)
            for s, a in zip(fold.test_stories, fold.test_actions, strict=True)
        }
        up, down, ids = _build_profile(fold)
        labels = np.asarray(fold.y_train, dtype=int)
        captured.append(
            Fold(
                fold=len(captured),
                candidates=[_card(s) for s in fold.candidates],
                labels=[judged.get(s.id, -1) for s in fold.candidates],
                production=[float(v) for v in result[0]],
                profile_up=up,
                profile_down=down,
                profile_ids=ids,
                train_counts={
                    "up": int((labels == UP).sum()),
                    "neutral": int((labels == NEUTRAL).sum()),
                    "down": int((labels == DOWN).sum()),
                },
            )
        )
        return result

    original_loader = evaluator.load_production_candidate_stories
    dropped: list[int] = []

    def loader(
        db: Database,
        config: Config,
        *,
        user_id: int | None,
        exclude_feedback: bool,
        now_ts: int | None = None,
        trace: RankTrace | _NullTrace = NULL_TRACE,
    ) -> list[Story]:
        """Drop candidates whose cached embedding is stale (text hash changed
        after embedding, re-embed pending); the canonical evaluator aborts on
        them.  Recorded in dropped_candidates.json."""
        stories = original_loader(
            db,
            config,
            user_id=user_id,
            exclude_feedback=exclude_feedback,
            now_ts=now_ts,
            trace=trace,
        )
        hashes = {
            s.id: h
            for s, h in zip(
                stories, evaluator._embedding_text_hashes(stories), strict=True
            )
        }
        cached = db.get_embeddings_batch(
            [s.id for s in stories], config.embedding_model_version, hashes
        )
        keep = [s for s in stories if s.id in cached]
        dropped.extend(s.id for s in stories if s.id not in cached)
        return keep

    with (
        patch.object(evaluator, "_production_scores", side_effect=score),
        patch.object(evaluator, "load_production_candidate_stories", loader),
    ):
        evaluator.main(
            [*evaluator_args, "--output", str(work / "evaluator_report.json")]
        )
    (work / "dropped_candidates.json").write_text(json.dumps(dropped) + "\n")
    print(f"dropped {len(dropped)} stale-embedding candidates: {dropped[:20]}")
    publish_immutable(
        work / "folds.json", json.dumps([asdict(f) for f in captured], indent=1) + "\n"
    )
    for f in captured:
        n_up = sum(label == UP for label in f.labels)
        n_down = sum(label == DOWN for label in f.labels)
        print(
            f"fold {f.fold}: {len(f.candidates)} candidates, held-out up={n_up} "
            f"down={n_down}; train {f.train_counts}; profile "
            f"{len(f.profile_up)} up / {len(f.profile_down)} down"
        )


def load_folds(work: Path, receipt: FrozenInputs | None = None) -> list[Fold]:
    import hashlib

    content = (work / "folds.json").read_bytes()
    if (
        receipt is not None
        and hashlib.sha256(content).hexdigest() != receipt.folds_sha256
    ):
        raise FrozenInputsError("folds.json differs from the initially frozen input")
    raw = json.loads(content)
    return [
        Fold(
            fold=f["fold"],
            candidates=[Card(**c) for c in f["candidates"]],
            labels=f["labels"],
            production=f["production"],
            profile_up=f["profile_up"],
            profile_down=f["profile_down"],
            profile_ids=f["profile_ids"],
            train_counts=f["train_counts"],
        )
        for f in raw
    ]


def state_for(fold: Fold, card: Card) -> dict[str, object]:
    story: dict[str, object] = {
        "title": card.title,
        "domain": card.domain or "(none)",
        "source": card.source,
    }
    if card.text:
        story["text_excerpt"] = card.text
    return {
        "reader_profile": {
            "recently_upvoted": fold.profile_up,
            "recently_downvoted": fold.profile_down,
        },
        "story": story,
    }


def request_body(fold: Fold, card: Card) -> dict[str, object]:
    return {"model": MODEL, "questions": QUESTION, "state": state_for(fold, card)}


def selected(fold: Fold, sample: int | None, seed: int) -> list[int]:
    """Candidate positions to score.

    Every judged (held-out voted) candidate is kept; ``sample`` caps the
    unjudged candidates with a uniform fixed-seed draw.  Judged-only metrics
    (AUC) are unaffected; top-k metrics shift level but stay paired.
    """
    if sample is not None and sample < 0:
        raise ValueError("sample size cannot be negative")
    judged = [i for i, label in enumerate(fold.labels) if label >= 0]
    unjudged = [i for i, label in enumerate(fold.labels) if label < 0]
    if sample is not None and sample < len(unjudged):
        unjudged = random.Random(seed * 1000 + fold.fold).sample(unjudged, sample)
    return sorted(judged + unjudged)


def _cache_path(work: Path, fold: int, story_id: int) -> Path:
    cache = "cache_strands" if BACKEND == "strands" else "cache"
    return work / cache / f"{fold}_{story_id}.json"


def strands_identity() -> CacheIdentity:
    import hashlib

    return CacheIdentity(
        backend="strands",
        model_id=STRANDS_MODEL,
        model_revision=STRANDS_REVISION,
        dtype=STRANDS_DTYPE,
        scorer_sha256=hashlib.sha256(STRANDS_WORKER.read_bytes()).hexdigest(),
    )


def checked_folds(
    work: Path, folds_arg: set[int] | None, receipt: FrozenInputs | None = None
) -> list[Fold]:
    folds = load_folds(work, receipt)
    numbers = [fold.fold for fold in folds]
    if len(numbers) != len(set(numbers)):
        raise ValueError("duplicate fold numbers")
    if folds_arg is not None and not folds_arg <= set(numbers):
        raise ValueError("requested fold is absent from the frozen manifest")
    folds = [fold for fold in folds if folds_arg is None or fold.fold in folds_arg]
    if not folds:
        raise ValueError("no evaluation folds selected")
    judged_ids: set[int] = set()
    for fold in folds:
        ids = [card.id for card in fold.candidates]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"fold {fold.fold}: empty or duplicate candidates")
        if len(ids) != len(fold.labels) or len(ids) != len(fold.production):
            raise ValueError(f"fold {fold.fold}: candidate/label/score shape mismatch")
        if any(
            type(label) is not int or label not in (-1, DOWN, NEUTRAL, UP)
            for label in fold.labels
        ):
            raise ValueError(f"fold {fold.fold}: invalid feedback class")
        if not np.isfinite(fold.production).all():
            raise ValueError(f"fold {fold.fold}: nonfinite production scores")
        if set(ids) & set(fold.profile_ids):
            raise ValueError(f"fold {fold.fold}: profile/candidate leakage")
        if not fold.profile_up and not fold.profile_down:
            raise ValueError(
                f"fold {fold.fold}: no real profile to contrast with empty"
            )
        judged = {
            sid for sid, label in zip(ids, fold.labels, strict=True) if label >= 0
        }
        if judged_ids & judged:
            raise ValueError(
                "repeated judged story across folds needs cluster-aware uncertainty"
            )
        judged_ids.update(judged)
    return folds


def profile_state(fold: Fold, card: Card, variant: str) -> dict[str, object]:
    state = state_for(fold, card)
    if variant == "empty":
        state["reader_profile"] = {"recently_upvoted": [], "recently_downvoted": []}
    elif variant != "real":
        raise ValueError(f"unknown profile variant: {variant}")
    return state


def strict_cache_path(
    work: Path, fold: Fold, card: Card, variant: str, identity: CacheIdentity
) -> Path:
    return cache_path(
        work / "cache_strands_v2" / variant,
        profile_state(fold, card, variant),
        QUESTION,
        identity,
    )


def prepare_strands_jobs(
    work: Path,
    sample: int | None = None,
    seed: int = 0,
    folds_arg: set[int] | None = None,
    snapshot: Path | None = None,
) -> Path:
    """Prepare the full paired scope; legacy score presence never removes jobs."""
    work = work.resolve()
    if sample is not None and sample < 0:
        raise ValueError("sample size cannot be negative")
    if snapshot is None:
        receipt = validate_frozen_inputs(work)
    else:
        receipt = freeze_inputs(work, snapshot)
    path, content, manifest = paired_plan(work, sample, seed, folds_arg, receipt)
    validate_frozen_inputs(work, snapshot)
    publish_immutable(path, content)
    publish_immutable(path.parent / "manifest.json", manifest)
    print(f"prepared immutable paired jobs: {path}")
    return path


def paired_plan(
    work: Path,
    sample: int | None,
    seed: int,
    folds_arg: set[int] | None,
    receipt: FrozenInputs,
) -> tuple[Path, str, str]:
    """Derive the entire plan from trusted inputs, including labels and scope."""
    import hashlib

    work = work.resolve()
    if sample is not None and sample < 0:
        raise ValueError("sample size cannot be negative")
    identity = strands_identity()
    jobs = []
    for fold in checked_folds(work, folds_arg, receipt):
        for i in selected(fold, sample, seed):
            card = fold.candidates[i]
            for variant in ("real", "empty"):
                state = profile_state(fold, card, variant)
                jobs.append(
                    {
                        "out": str(
                            strict_cache_path(work, fold, card, variant, identity)
                        ),
                        "group": f"{fold.fold}:{variant}",
                        "fold": fold.fold,
                        "story_id": card.id,
                        "variant": variant,
                        "state": state,
                        "question": QUESTION,
                        "cache_identity": asdict(identity),
                        "request_sha256": request_sha256(state, QUESTION, identity),
                    }
                )
    content = "".join(json.dumps(job, sort_keys=True) + "\n" for job in jobs)
    digest = hashlib.sha256(content.encode()).hexdigest()
    manifest = {
        "schema": 3,
        "jobs_sha256": digest,
        "frozen_inputs": asdict(receipt),
        "jobs": len(jobs),
        "stories_per_profile": len(jobs) // 2,
        "identity": asdict(identity),
        "sample_per_fold": sample,
        "seed": seed,
        "folds": sorted(folds_arg) if folds_arg is not None else None,
        "primary_blend_weight": PRIMARY_BLEND_WEIGHT,
        "primary_comparison": "real-minus-empty within-fold pooled judged-story AUC",
        "confirmation_used": False,
    }
    manifest_text = (
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    plan_digest = hashlib.sha256(manifest_text.encode()).hexdigest()
    path = work / "paired_plans_v3" / plan_digest / "jobs.jsonl"
    return path, content, manifest_text


def validate_paired_plan(
    work: Path,
    sample: int | None,
    seed: int,
    folds_arg: set[int] | None,
    snapshot: Path | None = None,
) -> tuple[FrozenInputs, Path]:
    receipt = validate_frozen_inputs(work, snapshot)
    path, content, manifest = paired_plan(work, sample, seed, folds_arg, receipt)
    verify_immutable(path, content)
    verify_immutable(path.parent / "manifest.json", manifest)
    return receipt, path


@dataclass(frozen=True)
class CoverageRow:
    fold: int
    profile: str
    expected: int
    valid: int
    missing: int
    invalid: int
    legacy_present_not_reused: int
    issues: list[str]


def audit_strands_cache(
    work: Path,
    sample: int | None,
    seed: int,
    folds_arg: set[int] | None,
    snapshot: Path | None = None,
) -> dict[str, object]:
    """Count complete, valid paired coverage before any efficacy metric is computed."""
    receipt, plan = validate_paired_plan(work, sample, seed, folds_arg, snapshot)
    identity = strands_identity()
    coverage: list[CoverageRow] = []
    for fold in checked_folds(work, folds_arg, receipt):
        positions = selected(fold, sample, seed)
        for variant in ("real", "empty"):
            valid = missing = invalid = legacy = 0
            issues = []
            for i in positions:
                card = fold.candidates[i]
                path = strict_cache_path(work, fold, card, variant, identity)
                if (
                    variant == "real"
                    and (
                        work / "cache_strands" / f"{fold.fold}_{card.id}.json"
                    ).exists()
                ):
                    legacy += 1
                if not path.exists():
                    missing += 1
                    continue
                try:
                    read_cached_score(
                        path,
                        profile_state(fold, card, variant),
                        QUESTION,
                        identity,
                        require_parity=True,
                    )
                    valid += 1
                except CacheValidationError as exc:
                    invalid += 1
                    if len(issues) < 5:
                        issues.append(str(exc))
            coverage.append(
                CoverageRow(
                    fold.fold,
                    variant,
                    len(positions),
                    valid,
                    missing,
                    invalid,
                    legacy,
                    issues,
                )
            )
    complete = all(row.missing == 0 and row.invalid == 0 for row in coverage)
    result = {
        "complete": complete,
        "identity": asdict(identity),
        "frozen_inputs": asdict(receipt),
        "plan": str(plan),
        "coverage": [asdict(row) for row in coverage],
        "primary_blend_weight": PRIMARY_BLEND_WEIGHT,
        "metrics_computed": False,
        "confirmation_used": False,
    }
    validate_paired_plan(work, sample, seed, folds_arg, snapshot)
    publish_result(work / "cache_audit_v3.json", json.dumps(result, indent=2) + "\n")
    return result


def cmd_estimate(
    work: Path, sample: int | None, seed: int, folds: set[int] | None
) -> None:
    if BACKEND == "strands":
        print("offline paired coverage; provider cost $0; inference not started")
        print(json.dumps(audit_strands_cache(work, sample, seed, folds), indent=2))
        return
    total_chars = 0
    calls = 0
    for fold in load_folds(work):
        if folds is not None and fold.fold not in folds:
            continue
        for i in selected(fold, sample, seed):
            card = fold.candidates[i]
            if _cache_path(work, fold.fold, card.id).exists():
                continue
            total_chars += len(json.dumps(request_body(fold, card)))
            calls += 1
    tokens = total_chars / 3.5  # conservative chars/token for JSON + titles
    print(
        f"uncached calls={calls} est_input_tokens={tokens:,.0f} "
        f"est_cost=${tokens * PRICE_PER_INPUT_TOKEN:.3f} (before calibration)"
    )


def _api_key() -> str:
    for line in ENV_PATH.read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip().strip("\"'")
    raise RuntimeError("OPENROUTER_API_KEY missing from env file")


def _http(url: str, key: str, body: dict[str, object] | None = None) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def _account_usage(key: str) -> float:
    return float(_http(CREDITS_URL, key)["data"]["total_usage"])


class Spend:
    """Spend ledger persisted in the work directory across reruns."""

    def __init__(self, work: Path, key: str) -> None:
        self.path = work / "spend.json"
        self.lock = threading.Lock()
        if self.path.exists():
            data = json.loads(self.path.read_text())
        else:
            data = {
                "baseline_account_usage": _account_usage(key),
                "reported_cost": 0.0,
                "input_tokens": 0,
                "calls": 0,
            }
        self.data: dict[str, float] = data
        self.save()

    def add(self, cost: float, tokens: int) -> None:
        with self.lock:
            self.data["reported_cost"] += cost
            self.data["input_tokens"] += tokens
            self.data["calls"] += 1

    def save(self) -> None:
        with self.lock:
            self.path.write_text(json.dumps(self.data, indent=1) + "\n")


def cmd_score(
    work: Path,
    sample: int | None,
    seed: int,
    folds: set[int] | None,
    cap: float,
    workers: int,
    limit: int | None,
) -> None:
    key = _api_key()
    spend = Spend(work, key)
    (work / "cache").mkdir(exist_ok=True)
    jobs: list[tuple[Fold, Card]] = []
    for fold in load_folds(work):
        if folds is not None and fold.fold not in folds:
            continue
        for i in selected(fold, sample, seed):
            card = fold.candidates[i]
            if not _cache_path(work, fold.fold, card.id).exists():
                jobs.append((fold, card))
    if limit is not None:
        jobs = jobs[:limit]
    print(f"{len(jobs)} uncached calls; cap ${cap:.2f}")
    stop = threading.Event()
    errors: list[str] = []

    def account_spent() -> float:
        return _account_usage(key) - spend.data["baseline_account_usage"]

    def run(job: tuple[Fold, Card]) -> None:
        if stop.is_set():
            return
        fold, card = job
        for attempt in range(4):
            try:
                response = _http(ENDPOINT, key, request_body(fold, card))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:300]
                if exc.code in (429, 500, 502, 503, 524, 529) and attempt < 3:
                    time.sleep(2**attempt)
                    continue
                errors.append(f"{fold.fold}/{card.id}: HTTP {exc.code} {detail}")
                if exc.code in (401, 402, 403):
                    stop.set()
                return
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < 3:
                    time.sleep(2**attempt)
                    continue
                errors.append(f"{fold.fold}/{card.id}: {exc!r}")
                return
        answer = response["answers"]["upvote"]
        p_yes = float(answer["noul"])
        if not 0.0 <= p_yes <= 1.0:
            raise ValueError(f"noul outside [0,1]: {answer}")
        usage = response.get("usage", {})
        tokens = int(usage.get("input_tokens", 0))
        cost = usage.get("cost")
        cost = float(cost) if cost is not None else tokens * PRICE_PER_INPUT_TOKEN
        spend.add(cost, tokens)
        _cache_path(work, fold.fold, card.id).write_text(
            json.dumps({"p_yes": p_yes, "response": response}) + "\n"
        )

    started = time.time()
    batch = 200
    for start in range(0, len(jobs), batch):
        spent = max(account_spent(), spend.data["reported_cost"])
        projected = (
            spent + (spend.data["reported_cost"] / max(spend.data["calls"], 1)) * batch
        )
        if projected >= cap:
            print(f"stopping: spent ${spent:.4f}, next batch would reach cap")
            break
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(run, jobs[start : start + batch]))
        spend.save()
        print(
            f"{min(start + batch, len(jobs))}/{len(jobs)} done, reported "
            f"${spend.data['reported_cost']:.4f}, account delta ${account_spent():.4f}, "
            f"{time.time() - started:.0f}s, errors {len(errors)}",
            flush=True,
        )
        if stop.is_set():
            break
    spend.save()
    for line in errors[:10]:
        print("error:", line)
    if errors:
        raise SystemExit(f"{len(errors)} calls failed")


def cmd_score_strands(
    work: Path,
    sample: int | None,
    seed: int,
    folds: set[int] | None,
    limit: int | None,
    python: str,
    threads: int,
    batch: int,
    snapshot: Path | None = None,
) -> None:
    """Score with Strands Decider locally: same state and question, no network."""
    jobs_path = prepare_strands_jobs(work, sample, seed, folds, snapshot)
    cmd = [
        python,
        str(STRANDS_WORKER),
        str(jobs_path),
        "--model",
        STRANDS_MODEL,
        "--model-revision",
        STRANDS_REVISION,
        "--threads",
        str(threads),
        "--batch",
        str(batch),
        "--dtype",
        STRANDS_DTYPE,
        "--check",
        "1",
        "--parity-tolerance",
        "0.0001",
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------- metrics


def _pct(scores: np.ndarray) -> np.ndarray:
    """Average-rank percentile in (0, 1]; ties share their mean rank."""
    order = scores.argsort(kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    for value in np.unique(scores):
        mask = scores == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks / len(scores)


def metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
    """Same definitions as eval_ranker_variants raw metrics (up = relevant)."""
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    n_up = int((labels == UP).sum())
    ranks = np.flatnonzero(ranked == UP)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(n_up, 40)))
    ndcg = sum(1 / math.log2(p + 2) for p in ranks if p < 40) / ideal if ideal else None
    top40 = ranked[:40]
    # AUC up vs rest of judged (down + neutral), ties counted one half.
    up_s = scores[labels == UP]
    other_s = scores[(labels == DOWN) | (labels == NEUTRAL)]
    if len(up_s) and len(other_s):
        greater = (up_s[:, None] > other_s[None, :]).sum()
        ties = (up_s[:, None] == other_s[None, :]).sum()
        auc: float | None = float((greater + 0.5 * ties) / (len(up_s) * len(other_s)))
    else:
        auc = None
    down_s = scores[labels == DOWN]
    if len(up_s) and len(down_s):
        g = (up_s[:, None] > down_s[None, :]).sum()
        t = (up_s[:, None] == down_s[None, :]).sum()
        auc_down: float | None = float((g + 0.5 * t) / (len(up_s) * len(down_s)))
    else:
        auc_down = None
    return {
        "ndcg_at_40": ndcg,
        "p_at_40": float((top40 == UP).sum() / len(top40)) if len(top40) else None,
        "down_at_40": float((top40 == DOWN).sum() / len(top40)) if len(top40) else None,
        "up_at_12": float((ranked[:12] == UP).sum()),
        "auc_up_vs_rest": auc,
        "auc_up_vs_down": auc_down,
    }


METRIC_NAMES = (
    "ndcg_at_40",
    "p_at_40",
    "down_at_40",
    "up_at_12",
    "auc_up_vs_rest",
    "auc_up_vs_down",
)
WEIGHTS = (0.1, 0.2, 0.3, 0.5)


def _sign_test(diffs: list[float]) -> float:
    """Exact two-sided binomial sign test p-value (zeros dropped)."""
    pos = sum(d > 0 for d in diffs)
    neg = sum(d < 0 for d in diffs)
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def _bootstrap_ci(
    diffs: list[float], seed: int = 0, reps: int = 10_000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(diffs)
    means = arr[rng.integers(0, len(arr), (reps, len(arr)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _story_bootstrap_auc(
    per_fold: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    seed: int = 0,
    reps: int = 2000,
) -> tuple[float, float, float]:
    """Pooled AUC(up vs rest) difference B - A, resampling judged stories within folds."""
    rng = np.random.default_rng(seed)

    def pooled(sel: list[np.ndarray]) -> float:
        num = den = 0.0
        for (a, b, labels), idx in zip(per_fold, sel, strict=True):
            lab = labels[idx]
            for s, sign in ((a[idx], -1.0), (b[idx], 1.0)):
                up = s[lab == UP]
                other = s[lab != UP]
                g = (up[:, None] > other[None, :]).sum() + 0.5 * (
                    up[:, None] == other[None, :]
                ).sum()
                num += sign * g
            den += (lab == UP).sum() * (lab != UP).sum()
        return num / den if den else 0.0

    judged = [np.flatnonzero(labels >= 0) for _, _, labels in per_fold]
    point = pooled(judged)
    draws = [pooled([rng.choice(j, len(j)) for j in judged]) for _ in range(reps)]
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def snapshot_popularity(
    snapshot: Path,
    story_ids: set[int],
    expected_sha256: str | None = None,
) -> dict[int, tuple[float, float | None]]:
    """Read only the immutable snapshot; missing values never become zero scores."""
    import sqlite3
    from contextlib import closing

    if not snapshot.is_file():
        raise ValueError("popularity snapshot is missing")
    for suffix in ("-wal", "-journal", "-shm"):
        sidecar = Path(str(snapshot) + suffix)
        if sidecar.exists() or sidecar.is_symlink():
            raise ValueError(
                "popularity snapshot has SQLite sidecars; require a standalone frozen snapshot"
            )

    def snapshot_stamp() -> tuple[int, int, int, int, int]:
        stat = snapshot.lstat()
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )

    before_stamp = snapshot_stamp()
    before = hash_file(snapshot)
    if expected_sha256 is not None and before != expected_sha256:
        raise FrozenInputsError(
            "popularity snapshot differs from frozen receipt at read time"
        )
    with closing(
        sqlite3.connect(snapshot.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    ) as conn:
        records = conn.execute("SELECT id, score, comment_count FROM stories")
        result: dict[int, tuple[float, float | None]] = {}
        for sid, points, comments in records:
            if sid not in story_ids:
                continue
            if (
                points is None
                or not math.isfinite(points)
                or (comments is not None and not math.isfinite(comments))
            ):
                raise ValueError(f"story {sid}: missing/nonfinite popularity metadata")
            result[sid] = (
                math.log1p(max(0, points)),
                math.log1p(max(0, comments)) if comments is not None else None,
            )
    # Restoring the original bytes does not restore inode identity or ctime.
    if hash_file(snapshot) != before or snapshot_stamp() != before_stamp:
        raise FrozenInputsError("popularity snapshot changed while reading")
    for suffix in ("-wal", "-journal", "-shm"):
        if Path(str(snapshot) + suffix).exists():
            raise FrozenInputsError(
                "popularity snapshot gained SQLite sidecars while reading"
            )
    if result.keys() != story_ids:
        raise ValueError("popularity snapshot does not cover the exact candidate pool")
    return result


def auc_comparison(
    paired: list[tuple[np.ndarray, np.ndarray, np.ndarray]], seed: int
) -> dict[str, object] | None:
    if not any(
        np.any(labels == UP) and np.any((labels == DOWN) | (labels == NEUTRAL))
        for _, _, labels in paired
    ):
        return None
    point, lo, hi = _story_bootstrap_auc(paired, seed=seed)
    return {"diff": point, "ci95": [lo, hi]}


def cmd_report(
    work: Path,
    sample: int | None,
    seed: int,
    folds_arg: set[int] | None,
    snapshot: Path | None = None,
) -> None:
    """Refuse partial or unverifiable scores before computing any efficacy metric."""
    work = work.resolve()
    if BACKEND != "strands":
        raise ValueError(
            "paired provenance contract is currently implemented only for Strands; legacy Decision-1 reports are disabled"
        )
    if snapshot is None:
        raise ValueError(
            "--snapshot is required for same-candidate popularity controls"
        )
    receipt = validate_frozen_inputs(work, snapshot, require_snapshot=True)
    audit = audit_strands_cache(work, sample, seed, folds_arg, snapshot)
    if not audit["complete"]:
        raise ValueError(
            "incomplete or invalid paired score coverage; see cache_audit_v3.json; no efficacy report computed"
        )
    folds = checked_folds(work, folds_arg, receipt)
    positions = [selected(fold, sample, seed) for fold in folds]
    popularity = snapshot_popularity(
        snapshot,
        {
            fold.candidates[i].id
            for fold, indexes in zip(folds, positions, strict=True)
            for i in indexes
        },
        receipt.snapshot_sha256,
    )
    missing_comments = sorted(
        sid for sid, (_, count) in popularity.items() if count is None
    )
    identity = strands_identity()
    rows: dict[str, list[dict[str, float | None]]] = {}
    paired: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    profile_paired: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    fold_info = []
    for fold, indexes in zip(folds, positions, strict=True):
        labels = np.asarray([fold.labels[i] for i in indexes])
        prod = np.asarray([fold.production[i] for i in indexes], dtype=float)
        model_scores = {}
        for variant in ("real", "empty"):
            model_scores[variant] = np.asarray(
                [
                    read_cached_score(
                        strict_cache_path(
                            work, fold, fold.candidates[i], variant, identity
                        ),
                        profile_state(fold, fold.candidates[i], variant),
                        QUESTION,
                        identity,
                        require_parity=True,
                    )
                    for i in indexes
                ]
            )
        variants = {
            "production": prod,
            "strands_real": model_scores["real"],
            "strands_empty": model_scores["empty"],
        }
        w = PRIMARY_BLEND_WEIGHT
        for variant in ("real", "empty"):
            variants[f"blend_{variant}_w{w}"] = (1 - w) * _pct(prod) + w * _pct(
                model_scores[variant]
            )
        variants["snapshot_points"] = np.asarray(
            [popularity[fold.candidates[i].id][0] for i in indexes], dtype=float
        )
        if not missing_comments:
            variants["snapshot_comments"] = np.asarray(
                [popularity[fold.candidates[i].id][1] for i in indexes], dtype=float
            )
        profile_paired.append((model_scores["empty"], model_scores["real"], labels))
        for name, scores in variants.items():
            rows.setdefault(name, []).append(metrics(scores, labels))
            paired.setdefault(name, []).append((prod, scores, labels))
        fold_info.append(
            {
                "fold": fold.fold,
                "scored_candidates": len(indexes),
                "held_out_up": int((labels == UP).sum()),
                "held_out_down": int((labels == DOWN).sum()),
                "held_out_neutral": int((labels == NEUTRAL).sum()),
            }
        )
    summary: dict[str, dict[str, object]] = {}
    for name, per_fold in rows.items():
        entry: dict[str, object] = {}
        for metric in METRIC_NAMES:
            values = [r[metric] for r in per_fold]
            base = [r[metric] for r in rows["production"]]
            ok = [
                (v, b)
                for v, b in zip(values, base, strict=True)
                if v is not None and b is not None
            ]
            diffs = [v - b for v, b in ok]
            entry[metric] = {
                "per_fold": values,
                "mean": float(np.mean([v for v, _ in ok])) if ok else None,
                "diff_vs_production": float(np.mean(diffs)) if diffs else None,
                "diff_ci95_fold_bootstrap": _bootstrap_ci(diffs, seed=seed)
                if diffs and name != "production"
                else None,
                "sign_test_p": _sign_test(diffs)
                if diffs and name != "production"
                else None,
            }
        if name != "production":
            entry["pooled_auc_up_vs_rest_diff_story_bootstrap"] = auc_comparison(
                paired[name], seed
            )
        summary[name] = entry
    out = {
        "model": STRANDS_MODEL,
        "cache_identity": asdict(identity),
        "folds_sha256": receipt.folds_sha256,
        "snapshot_sha256": receipt.snapshot_sha256,
        "frozen_inputs": asdict(receipt),
        "plan": audit["plan"],
        "sample_per_fold": sample,
        "seed": seed,
        "folds": fold_info,
        "summary": summary,
        "primary_blend_weight": PRIMARY_BLEND_WEIGHT,
        "primary_real_minus_empty_auc": auc_comparison(profile_paired, seed),
        "popularity_uses_snapshot_time_values_not_vote_time": True,
        "inference_scope": "exploratory frozen folds; confirmation set not accessed",
        "confirmation_used": False,
        "coverage_complete": True,
        "secondary_comparisons_are_exploratory": True,
        "unavailable_controls": {
            "snapshot_comments": {
                "reason": "NULL comment counts in selected candidate pool",
                "missing_story_ids": missing_comments,
                "missing_count": len(missing_comments),
                "no_candidate_dropping_or_imputation": True,
            }
        }
        if missing_comments
        else {},
    }
    final_receipt, _ = validate_paired_plan(work, sample, seed, folds_arg, snapshot)
    if final_receipt != receipt or read_frozen_inputs(work) != receipt:
        raise FrozenInputsError("frozen input receipt changed during reporting")
    publish_result(
        work / "report_strands_v3.json",
        json.dumps(out, indent=2, allow_nan=False) + "\n",
    )
    print(
        "complete paired report: report_strands_v3.json; fixed blend weight",
        PRIMARY_BLEND_WEIGHT,
    )


def _fold_set(text: str | None) -> set[int] | None:
    return {int(x) for x in text.split(",")} if text else None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "phase", choices=("folds", "estimate", "prepare", "audit", "score", "report")
    )
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Immutable frozen SQLite snapshot for popularity controls",
    )
    parser.add_argument(
        "--sample-per-fold", type=int, help="Cap on unjudged candidates per fold"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", help="Comma-separated fold numbers (default all)")
    parser.add_argument(
        "--cap", type=float, default=1.80, help="Stop before spend reaches this ($)"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--limit", type=int, help="Score at most this many calls (smoke)"
    )
    parser.add_argument(
        "--backend", choices=("decision1", "strands"), default="decision1"
    )
    parser.add_argument(
        "--strands-python",
        default="/tmp/strands-eval/.venv/bin/python",
        help="Python of the venv with strands-decider installed",
    )
    parser.add_argument("--threads", type=int, default=3, help="strands torch threads")
    parser.add_argument("--batch", type=int, default=4, help="strands batch size")
    args, evaluator_args = parser.parse_known_args(argv)
    args.work = args.work.resolve()
    if evaluator_args and args.phase != "folds":
        parser.error("unexpected arguments: " + " ".join(evaluator_args))
    global BACKEND
    BACKEND = args.backend
    folds = _fold_set(args.folds)
    if args.sample_per_fold is not None and args.sample_per_fold < 0:
        parser.error("--sample-per-fold must be nonnegative")
    if args.phase in ("prepare", "audit") and BACKEND != "strands":
        parser.error("prepare/audit require --backend strands")
    if args.phase == "prepare":
        prepare_strands_jobs(
            args.work, args.sample_per_fold, args.seed, folds, args.snapshot
        )
    elif args.phase == "audit":
        result = audit_strands_cache(
            args.work, args.sample_per_fold, args.seed, folds, args.snapshot
        )
        print(json.dumps(result, indent=2))
        if not result["complete"]:
            raise SystemExit("incomplete paired coverage; no metrics computed")
    elif args.phase == "folds":
        rest = evaluator_args[1:] if evaluator_args[:1] == ["--"] else evaluator_args
        cmd_folds(args.work, rest)
    elif args.phase == "estimate":
        cmd_estimate(args.work, args.sample_per_fold, args.seed, folds)
    elif args.phase == "score" and BACKEND == "strands":
        cmd_score_strands(
            args.work,
            args.sample_per_fold,
            args.seed,
            folds,
            args.limit,
            args.strands_python,
            args.threads,
            args.batch,
            args.snapshot,
        )
    elif args.phase == "score":
        cmd_score(
            args.work,
            args.sample_per_fold,
            args.seed,
            folds,
            args.cap,
            args.workers,
            args.limit,
        )
    else:
        cmd_report(args.work, args.sample_per_fold, args.seed, folds, args.snapshot)


if __name__ == "__main__":
    main()
