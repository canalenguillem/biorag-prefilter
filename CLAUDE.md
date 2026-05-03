# CLAUDE.md — Working Context for Claude Code

This file is read at the start of every Claude Code session in this repo. Keep it short, current, and decisive.

## Project

BioRAG: a retrieval system that combines structured filtering (Polars on Parquet) with semantic search (Qdrant) over the UCI Drug Review dataset (~215k patient reviews from drugs.com). Full architecture and reasoning in `README.md` — read it first.

## Architecture in one paragraph

A natural-language question hits a FastAPI endpoint. An LLM extracts structured `Filters` (Pydantic). Polars (lazy) applies those filters and any windowed aggregations to the Parquet store, returning a list of review IDs. Qdrant runs vector search constrained to that ID set. A second LLM call composes the final answer from the retrieved chunks. The frontend (React + Vite + TS) exposes each stage of the pipeline as a debug panel.

The canonical regression query is the **rating-drop + fatigue query** described in the README. Every layer must support it. When in doubt about a design decision, ask: "does this make the killer query better, easier to test, or easier to explain?"

## Stack

- **Python 3.11+**, `pip + venv + requirements.txt` (no uv, no poetry, no conda)
- **Polars** (lazy by default) on **Parquet**
- **Qdrant** in Docker, accessed via the official Python client
- **OpenAI** `text-embedding-3-small` for embeddings
- **Anthropic Claude** for filter extraction and answer composition, via structured output / tool use
- **FastAPI** + **pydantic-settings** for config
- **React + Vite + TypeScript** for the demo UI, Tailwind for minimal styling
- **Docker Compose** orchestrates `qdrant`, `api`, `web`

## Code conventions

- All code, comments, identifiers, and docstrings in **English**.
- **Type hints on every public function**, including return types.
- **Polars is lazy-first.** Use `pl.scan_parquet`, build expressions, call `.collect()` only at the boundary of a function returning concrete data. If `.collect()` appears mid-pipeline, stop and reconsider.
- **No pandas.** If a Polars idiom is unclear, ask before reaching for an alternative.
- **Pure functions for filters.** `filters.py` exposes functions of shape `(LazyFrame, Filters) -> LazyFrame`. No I/O inside.
- **No print debugging in committed code.** Use the `logging` module, configured once in `config.py`.
- **One responsibility per module** — respect the layout in the README. Don't merge layers.
- **Tests next to behavior, not coverage targets.** A regression test for the killer query is worth more than fifty trivial ones.

## Working with the author

The author is Guillem — vocational training informatics teacher, 20+ years in tech, fluent in Python / Docker / FastAPI / Qdrant. **New on this project: Polars.** This repo is both a learning exercise and an interview portfolio piece for MiLaboratories (Polars + DataFusion shop in Bilbao). Adjust accordingly:

- **When introducing a new Polars pattern, explain it before writing it.** Don't drop a 50-line expression chaining `over()`, `rolling()`, and `when/then/otherwise` without naming each piece.
- **Prefer to scaffold a function and let the author write the Polars expression.** The queries are the part that matters in the interview. Where the author asks for a full implementation, write it — but flag the patterns being used.
- **Don't introduce libraries not in `requirements.txt` without flagging it explicitly** and waiting for a yes.
- **Don't expand scope.** If a feature seems useful, add it to the roadmap in `README.md` or as a `TODO` comment — don't build it.
- **Communication style: direct, no excessive preamble, no padding.** The author values honest pushback when a request is wrong or under-specified.
- **Spanish/Catalan in chat is fine** when the author writes in those languages, but all artifacts (code, comments, commit messages, docstrings, UI strings) stay in English.

## Out of scope (do not build unless explicitly requested)

- Rerankers, hybrid BM25, query rewriting
- Caching layers (Redis, in-memory)
- Authentication, rate limiting, user accounts
- Production deployment concerns (k8s, secrets management beyond `.env`)
- An evaluation harness beyond a handful of regression queries

These omissions are deliberate and documented in `README.md` ("What is not here"). Surface them as roadmap items, never as silent additions.

## Commands

```bash
# Backend dev environment
cd backend
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run a pipeline script
python scripts/01_ingest.py

# Tests
pytest tests/ -v

# Frontend dev
cd frontend
npm install
npm run dev

# Full stack
docker compose up
```

## Important files

- `README.md` — architecture, decisions, roadmap (read first, every session)
- `backend/src/biorag/filters.py` — heart of the structured layer
- `backend/src/biorag/extractor.py` — LLM/Polars contract via Pydantic
- `backend/src/biorag/retriever.py` — orchestrates filter → vector search

## Build state

The roadmap order below is fixed. Do not skip ahead. Update this checklist when a step lands on `main`.

- [ ] 1. Ingest — CSV → Parquet, schema verified
- [ ] 2. Filters — pure Polars functions + unit tests
- [ ] 3. Embeddings + Qdrant upsert (idempotent, resumable)
- [ ] 4. Extractor — Pydantic schema + Claude tool use
- [ ] 5. Retriever — glue layer, Polars → Qdrant id contract verified
- [ ] 6. Answer composer
- [ ] 7. FastAPI endpoint exposing every stage
- [ ] 8. Frontend — 3-panel debug UI