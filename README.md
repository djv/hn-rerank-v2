# HN Rerank

A personal Hacker News, Reddit and RSS reranker trained on each profile's votes.
It stores stories and feedback in SQLite, computes embeddings locally, and
serves ranked decks through a web dashboard and a Textual terminal reader.
The terminal reader also offers a Queue that mixes the server's ranked views.
A new profile starts with popularity-based ranking before it has enough votes.

Illustrative text mock; titles below are examples, not a captured session:

```text
HN Rerank                  Recommended | Popular | Explore
Window: 1 week             ↑ up   · neutral   ↓ down
> SQLite and the local-first application       [HN]
  A discussion of dataset provenance           [RSS]
  Summary / discussion / Why this story
```

## Architecture

```text
HN (ClickHouse + item fallback), Reddit (Arctic Shift/RSS), RSS
  -> ingestion and comment/article enrichment
  -> SQLite: stories, feedback, cached text, embeddings, interaction ledger
  -> local ONNX embeddings (mxbai-embed-xsmall-v1; optional Gemma side vectors)
  -> per-profile ranking model -> deck assembly, deduplication, exploration
  -> Flask server -> web dashboard + JSON API -> Textual TUI
Votes return to SQLite and train the next personalized ranking.
```

The checked-in deployment uses a joined logistic classifier over embeddings,
metadata and TF-IDF words. The earlier SVM/linear blend remains available.
Summary generation uses external LLM APIs; ranking and embeddings run locally.
See [architecture](ARCHITECTURE.md) and [configuration](config.example.toml).

## Evaluation and limits

[Evaluation summary](docs/EVALUATION.md) describes chronological held-out votes,
URL-group isolation, model choices and live outcomes. A separate time-split
input study reported held-out AUC about 0.78; joined logistic beat the earlier
blend's mean AUC in an exploratory study but put more downvotes near the top.
The live interleaving trial ended early without a statistically established win.

Judged-only evaluation ranks previously rated stories; it does not reconstruct
historical exposure. Current stored text and engagement can postdate the vote,
and semantic duplicates at different URLs can remain. Results concern one
profile and do not establish benefit for other profiles or a win over HN.
See [evaluation modes](docs/RANKER_EVALUATION.md) for commands and safeguards.

## Data and privacy

Stored data includes profile bearer tokens, up/neutral/down votes, seen state,
impressions, opens, dwell, deck scores, trial attribution and performance records.
Votes overwrite earlier labels; undo deletes a vote, not its interaction history.
`prune_stories` is dormant: its 60-day default is not an active retention policy.
Telemetry has no retention job yet. There is no account erasure or export API.
Profile deletion requires operator cleanup of database rows, derived model caches,
client credentials and backups. Clearing a cookie only creates a new profile.
The backup script keeps 30 snapshots by default; backups can include `.env`.
See [stored data, deletion limits and external transfers](docs/PRIVACY.md).

## Run from a fresh clone

Python 3.12+ and [uv](https://docs.astral.sh/uv/) are required. From the repo root:

```sh
uv sync
cp config.example.toml config.toml
uv run python setup_model.py
uv run python server.py
```

Open `http://127.0.0.1:8766/`. In another terminal, run
`uv run hn-rerank --server http://127.0.0.1:8766/` for the TUI.
Model setup downloads model/tokenizer files and CSS. Ingestion needs network access;
provider keys are optional for ranking and required for generated summaries.
Keep keys in an untracked `.env`; see [.env.example](.env.example).
[Deployment notes](docs/DEPLOYMENT.md) cover shared paths and existing VPS settings.

## Checks

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run pytest tests -n 2
uv run ruff check .
uv run ty check
uv build --package hn-rerank
```

The default backend suite excludes slow/browser tests and reduces local Hypothesis
budgets. Real-model tests skip locally if the model is absent; CI provisions it.
TUI tests run separately: `uv run pytest clients/tui/tests -n 2`.
Browser tests need system Chrome and the `browser` dependency group:
`uv run --group browser pytest tests/test_browser.py -m browser`.
See the [terminal client guide](clients/tui/README.md) for client details.

## Repo map

- `pipeline/`, `database.py`, `ch_client.py`: ranking, storage and ingestion.
- `server.py`, `templates/`, `clients/tui/`: service, dashboard and terminal client.
- `scripts/`, `tests/`: tools and regression tests.
- `docs/EVALUATION.md`: aggregate results, methodology and limitations.
- `ARCHITECTURE.md`, `ROADMAP.md`: system reference and historical roadmap.

Licensed under the [MIT License](LICENSE).
