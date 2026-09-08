"""CLI boot smoke tests: every argparse entry point must survive --help.

Guards the bug class where a malformed help string (e.g. a bare `%`)
crashes argument formatting before the command runs. Only scripts that
call ``parse_args`` are listed: the remaining mains take no CLI flags,
so invoking them would *run* the script instead of printing help.
All listed mains build their parser before any side effect (audited),
so ``--help`` always exits during argument parsing.
"""

import importlib
import subprocess
import sys
from unittest.mock import patch

import pytest

SCRIPT_MAINS = [
    "backfill_hn_comments",
    "bakeoff_embedding_models",
    "bakeoff_tldr_providers",
    "bench_qwen_embed_speed",
    "benchmark_embeddings",
    "benchmark_precomputed_svm",
    "benchmark_rank_cold_cache",
    "cap_loss_check",
    "cap_sweep",
    "deck_composition_report",
    "drop_dead_tables",
    "ledger_report",
    "migrate_db_to_strict",
    "migrate_interaction_events",
    "narrowing_report",
    "perf_report",
    "prepare_embedding_cache",
    "remove_source_stories",
    "seed_hn_from_bq",
    "seed_hn_from_clickhouse",
    "seed_smoke_test",
]


@pytest.mark.parametrize("module_name", SCRIPT_MAINS)
def test_cli_help_exits_zero(module_name: str) -> None:
    module = importlib.import_module(f"scripts.{module_name}")
    with pytest.raises(SystemExit) as error:
        with patch.object(sys, "argv", [module_name, "--help"]):
            module.main()
    assert error.value.code == 0


def test_eval_cli_help_exits_zero() -> None:
    from scripts import eval_ranker_variants

    with pytest.raises(SystemExit) as error:
        eval_ranker_variants.main(["--help"])
    assert error.value.code == 0


def test_top_level_migrate_help_exits_zero() -> None:
    completed = subprocess.run(
        [sys.executable, "migrate_feedback.py", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0
    assert "usage:" in completed.stdout
