# DaniBookLM

100% local, offline, privacy-first clone of **Google NotebookLM** powered by open-weight **Gemma 4 Multimodal** models on Apple Silicon (Metal) and Windows/Linux (CUDA/CPU).

## Architecture & Domain

- **Two-Tier Memory ([ADR 0001](docs/adr/0001-two-tier-local-architecture.md))**:
  - **Desk (Tier 1 — `src/danibook/memory/tier1_desk.py`)**: Dynamic prefix budget keeping active Sources in narrative focus.
  - **Archive (Tier 2 — `src/danibook/memory/tier2_archive.py`)**: Persistent SQLite FTS5 BM25 index (`data/cache/archive_fts5.sqlite`) with instant `SHA-256` startup caching and inline Citations (`[Source, Page/Section]`).
- **Hybrid Bun + Metal Runtime ([ADR 0002](docs/adr/0002-bun-hybrid-architecture.md))**:
  - **Bun Core Gateway (`src/bun/server.ts`, port `7861`)**: High-throughput static server, native `bun:sqlite` FTS5 search, and SSE reverse proxy.
  - **Metal Inference Daemon (`src/danibook/ui/app.py`, port `7860`)**: Thread-safe `llama-cpp-python` + `Gemma4ChatHandler` (`mmproj`) for text and vision synthesis, with on-demand Hugging Face downloads, progress tracking, VRAM unloading, and disk cleanup.
- **Domain Layer (`src/danibook/domain/`)**:
  - `NotebookService`, `StudioStore`, and `CatalogRepository` cleanly decouple domain orchestration from HTTP transport.
- **Material Design 3 UI (`src/danibook/ui/notebooklm.html`)**:
  - Official `@material/web` components, 3-state theme switcher (`System` default, `Light`, `Dark`), 5 curated **Collections**, **Sample** picker, multimodal chat with image inspection, and 4 **Studio Artifacts** (*Guía de Estudio*, *Mapa Conceptual / Glosario*, *Cronología*, *Citas Textuales*).

## Quickstart

```bash
# 1. Start the Metal Inference Daemon (port 7860)
/opt/homebrew/bin/uv run python -m danibook.ui.app

# 2. Start the Bun Core Gateway (port 7861)
~/.bun/bin/bun src/bun/server.ts

# 3. Run the test suite
/opt/homebrew/bin/uv run pytest tests/
```

## Governance & Documentation

- [AGENTS.md](AGENTS.md) — Agent operating loop and context pointers.
- [GLOSSARY.md](GLOSSARY.md) — Canonical domain vocabulary.
- [CONSTITUTION.md](CONSTITUTION.md), [MISSION.md](MISSION.md), [VALUES.md](VALUES.md), [VISION.md](VISION.md) — Core invariants and tenets.
- [docs/adr/](docs/adr) — Architecture Decision Records.
- [docs/research/](docs/research) — Technical spikes and empirical findings.
