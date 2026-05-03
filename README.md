# BioRAG — Pre-filtered Retrieval over Patient Drug Reviews

A retrieval system that combines structured filtering (Polars over Parquet) with semantic search (Qdrant) over the UCI Drug Review dataset (~215k patient-written reviews from drugs.com). Given a natural-language question, an LLM extracts structured filters, Polars narrows the candidate set against metadata and aggregations, and Qdrant performs vector search only over what remains. A second LLM call composes the answer.

## What it answers

Example queries the system is designed to handle:

- *"What do patients say about side effects of sertraline reviewed after 2016 with rating ≥ 8?"*
- *"Among antidepressants whose average rating dropped by more than 1.5 points in the last 12 months compared to the prior year, which ones do reviewers most often associate with fatigue?"*
- *"Top-rated medications for migraine where reviewers report rapid onset of relief."*

The first one is filterable in any vector store. The second is the one that justifies the architecture — see below.

## Why pre-filtering

A vanilla RAG embeds every document and asks the vector store for the top-k against a query embedding. That is the right choice when the search space is small, the filters are simple, and the question is plainly semantic.

Three things break that model:

1. **Aggregations and time windows.** "Average rating in the last 12 months versus the year before" is not metadata — it is *computed* metadata. Vector stores do not run group-bys.
2. **Cheap structured constraints over a large corpus.** When the metadata can eliminate 90% of rows for free, running vector search over the full 215k is wasted compute and dilutes top-k quality.
3. **Reproducibility and inspectability.** A Polars expression is auditable; an embedding similarity score is not. For biomedical work this matters.

The pattern in this repo addresses all three by placing a structured layer in front of the vector layer.

## Architecture

```
        ┌──────────────────────┐
        │  natural-language Q  │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  LLM extractor       │   structured output → Pydantic
        │  question → Filters  │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Polars (lazy)       │   filters + windowed aggregations
        │  on Parquet          │   → list[review_id]
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Qdrant              │   vector search constrained
        │  top-k chunks        │   by the id set
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  LLM answerer        │
        │  context → answer    │
        └──────────────────────┘
```

The contract between layers is explicit: a Pydantic schema between the LLM extractor and Polars, a primary-key list between Polars and Qdrant, the retrieved chunks between Qdrant and the answering LLM. Each layer is testable in isolation.

## The query that justifies the architecture

> *"Among antidepressants whose average rating dropped more than 1.5 points in the last 12 months compared to the prior year, find reviews where patients mention fatigue."*

Why this is hard for any single tool:

- **Pure Qdrant** cannot compute the rating drop. Payload filters cover equality, ranges, and set membership — not group-bys with windowed aggregations.
- **Pure Polars** cannot find "mentions fatigue". A naive substring match misses *exhausted*, *no energy*, *I sleep all day*, *like a zombie*.
- **Polars first, then Qdrant** computes the drop server-side, returns ~150 drug names, narrows the corpus to the ~3000 reviews of those drugs, and runs vector search only over that subset.

This is the demo query for interviews and the canonical regression test for the system.

## Stack decisions

**Polars (lazy) over pandas / DuckDB / SQL.** Polars matches the target stack, expresses windowed aggregations cleanly, and the lazy API exposes the optimizer's plan — useful for explaining the system. DuckDB would be a defensible alternative; Polars chosen for stack alignment.

**Qdrant over pgvector / Weaviate / Chroma.** Already familiar, trivial to run in Docker, payload filtering exists for simple post-filter checks, and the `HasId` condition lets us push a Polars-derived id list straight into the search request.

**OpenAI `text-embedding-3-small`.** Roughly $0.6 to embed the full 215k once. Cheap enough that re-embedding is not an architectural commitment.

**Claude for filter extraction and answering.** Tool use / structured output is reliable, and per-query cost is negligible at this scale.

**FastAPI.** A single `POST /query` endpoint exposes the full flow and returns every intermediate stage, not just the final answer.

**React + Vite + TS frontend, deliberately minimal.** One page that exposes the three internal stages of the pipeline (extracted filters, Polars row count and ids, Qdrant chunks, final answer). The goal is to make the architecture *visible*, not to ship product UI.

**Docker Compose.** Three services — `qdrant`, `api`, `web` — so the project runs with one command.

## What is not here (and why)

No reranker, no hybrid BM25, no query rewriting, no caching, no auth, no eval harness beyond a handful of regression queries. All sensible next steps; none of them is the point of this MVP. The point is the layered filter→search→answer pattern. Adding more before the pattern is solid would be premature.

## Repo layout

```
biorag-prefilter/
├── backend/
│   ├── requirements.txt
│   ├── src/biorag/
│   │   ├── config.py        # pydantic-settings
│   │   ├── ingest.py        # CSV → Polars → Parquet
│   │   ├── embed.py         # embeddings → Qdrant upsert
│   │   ├── filters.py       # Polars filter functions
│   │   ├── extractor.py     # LLM: NL question → Filters (Pydantic)
│   │   ├── retriever.py     # orchestrates filter + vector search
│   │   ├── answer.py        # LLM: context → final answer
│   │   └── api.py           # FastAPI app
│   ├── scripts/
│   │   ├── 01_ingest.py
│   │   ├── 02_embed.py
│   │   └── 03_query_cli.py  # end-to-end without the API
│   └── tests/
│       └── test_filters.py
├── frontend/
│   ├── package.json
│   └── src/
│       ├── App.tsx          # input + 3-stage debug panel
│       └── main.tsx
├── data/                    # gitignored
│   ├── raw/
│   └── processed/
├── docker-compose.yml
├── .env.example
├── CLAUDE.md                # working context for Claude Code sessions
└── README.md
```

## Build order

Each step is a thin, testable slice:

1. **Ingest.** `scan_csv` → date parsing → Parquet write. Verify schema and row count.
2. **Filters.** A small set of pure functions over `LazyFrame`, with unit tests on a fixture.
3. **Embedding + Qdrant upsert.** The only step that costs money; idempotent and resumable.
4. **Extractor.** Pydantic schema, structured output, golden-set tests on canonical questions.
5. **Retriever.** Glue layer; verify the id-list contract from Polars to Qdrant.
6. **Answer composer.** Short, honest prompts; cite the chunks used.
7. **FastAPI.** Single endpoint that returns every stage of the pipeline.
8. **Frontend.** One page, three debug panels.

## Running it

```bash
cp .env.example .env             # add API keys
docker compose up -d qdrant
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/01_ingest.py
python scripts/02_embed.py
python scripts/03_query_cli.py "your question here"
docker compose up                # full stack with frontend
```

(Commands above are the target shape; some scripts are stubs until the corresponding build step lands.)

## A note on tooling

This project is developed using Claude Code as a pair-programming tool inside VS Code. Architectural decisions, the Polars query design, the schema contracts, and the choice of test cases are the author's; Claude Code primarily handles integration plumbing and boilerplate. The companion file `CLAUDE.md` documents the working context provided to each session.