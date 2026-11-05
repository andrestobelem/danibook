# AGENTS.md

**DaniBookLM** — Local, offline NotebookLM environment powered by quantized Gemma 4 models on Apple Silicon Metal and a hybrid Bun + Python runtime.

## Context Pointers

### Governance & Domain
- [GLOSSARY.md](file:///Users/andrestobelem/ws/at/danibook/GLOSSARY.md): **Canonical Language** — Naming classes, endpoints, UI labels, or domain concepts (`Notebook`, `Source`, `Desk`, `Archive`, `Collection`, `Sample`, `Citation`, `Studio`, `Studio Artifact`, `Concept Map / Glossary`, `Note`, `Deep Research`).
- [CONSTITUTION.md](file:///Users/andrestobelem/ws/at/danibook/CONSTITUTION.md): **Invariants & Escalation** — Hitting architectural boundaries, missing credentials, or conflicting specs.
- [VALUES.md](file:///Users/andrestobelem/ws/at/danibook/VALUES.md): **Core Tenets** — Weighing speed against empirical proof, or scope against refactoring.
- [MISSION.md](file:///Users/andrestobelem/ws/at/danibook/MISSION.md): **Litmus Test** — Judging whether an intervention adds leverage or churn.
- [VISION.md](file:///Users/andrestobelem/ws/at/danibook/VISION.md): **BHAG** — Long-horizon zero-hallucination autonomy targets.
- [README.md](file:///Users/andrestobelem/ws/at/danibook/README.md): **Quickstart & Overview** — Updating startup commands, ports, or user-facing documentation.

### Architecture & Research
- [docs/adr/0001-two-tier-local-architecture.md](file:///Users/andrestobelem/ws/at/danibook/docs/adr/0001-two-tier-local-architecture.md): **Two-Tier Memory** — Modifying `Desk` (Tier 1 continuous prefix budget) or `Archive` (Tier 2 SQLite FTS5 BM25 index).
- [docs/adr/0002-bun-hybrid-architecture.md](file:///Users/andrestobelem/ws/at/danibook/docs/adr/0002-bun-hybrid-architecture.md): **Hybrid Runtime** — Altering the seam between the Bun Core Gateway (`:7861`) and the Python Metal Inference Daemon (`:7860`).
- [docs/research/2026-11-03-gemma4-local-notebooklm.md](file:///Users/andrestobelem/ws/at/danibook/docs/research/2026-11-03-gemma4-local-notebooklm.md): **Gemma 4 & Memory Findings** — Adjusting KV cache stability, OCR control-byte sanitization, or context window limits.
- [docs/research/2026-11-03-bun-migration-spike.md](file:///Users/andrestobelem/ws/at/danibook/docs/research/2026-11-03-bun-migration-spike.md): **Bun & SQLite Benchmarks** — Changing `Bun.serve`, SSE proxying, or `bun:sqlite` FTS5 queries.

### Skills & Reference Formats
- [`design-principles`](file:///Users/andrestobelem/ws/at/danibook/.agents/skills/design-principles/SKILL.md) (`.agents/skills/design-principles/SKILL.md`): **Component Design** — Enforcing modularity, high cohesion, separation of responsibilities, abstraction, and low coupling when structuring or refactoring modules.
- [`domain-modeling`](file:///Users/andrestobelem/ws/at/danibook/.agents/skills/domain-modeling/SKILL.md) (`.agents/skills/domain-modeling/SKILL.md`): **Ubiquitous Language & Decisions** — Challenging fuzzy terminology, stress-testing domain boundaries with concrete scenarios, and maintaining canonical terms:
  - [`GLOSSARY-FORMAT.md`](file:///Users/andrestobelem/ws/at/danibook/.agents/skills/domain-modeling/GLOSSARY-FORMAT.md): Structure and rules for `GLOSSARY.md` (`_Avoid_` lists, zero implementation leakage).
  - [`ADR-FORMAT.md`](file:///Users/andrestobelem/ws/at/danibook/.agents/skills/domain-modeling/ADR-FORMAT.md): One-paragraph template and 3-part test (hard to reverse, surprising without context, real trade-off) for `docs/adr/`.
- [`grilling`](file:///Users/andrestobelem/ws/at/danibook/.agents/skills/grilling/SKILL.md) (`.agents/skills/grilling/SKILL.md`): **Plan & Assumption Stress-Testing** — Probing edge cases, trade-offs, and hidden assumptions before committing to an implementation plan.
- [`writing-for-agents`](file:///Users/andrestobelem/ws/at/danibook/.agents/skills/writing-for-agents/SKILL.md) (`.agents/skills/writing-for-agents/SKILL.md`): **Agent Documentation & Pruning** — Shaping context pointers, progressive disclosure, positive framing, checkable completion criteria, and pruning no-ops:
  - [`SKILL-MECHANICS.md`](file:///Users/andrestobelem/ws/at/danibook/.agents/skills/writing-for-agents/SKILL-MECHANICS.md): Frontmatter, invocation control, and skill routing mechanics.

## System Map & In-File Rules

- **Domain Layer** ([src/danibook/domain/](file:///Users/andrestobelem/ws/at/danibook/src/danibook/domain)): `NotebookService`, `StudioStore`, and `CatalogRepository` own all business rules, filesystem writes, and multimodal attachment gating (`is_active_model_multimodal`). Keep HTTP routers in [src/danibook/ui/app.py](file:///Users/andrestobelem/ws/at/danibook/src/danibook/ui/app.py) as thin transport adapters.
- **Model Lifecycle & Multimodal Gating** ([src/danibook/llm/engine.py](file:///Users/andrestobelem/ws/at/danibook/src/danibook/llm/engine.py), [src/danibook/config.py](file:///Users/andrestobelem/ws/at/danibook/src/danibook/config.py)):
  - Switching models unloads the previous model from Metal VRAM (`unload_active_model()`) before loading or downloading the target GGUF/mmproj from Hugging Face with real-time progress (`/api/models/progress`).
  - Models can be deleted from disk (`DELETE /api/models/delete`) and re-downloaded on demand.
  - Multimodal attachments (images/audio in chat and `Source` uploads) are enabled exclusively when the active model preset has `supports_vision=True`; text-only models restrict uploads and inputs to text documents.
  - Native `llama.cpp` calls run inside `self._lock` (`threading.Lock()`), and `generate_chat_stream` evaluates remaining context headroom (`evaluate_context_headroom`), automatically compacts older conversation turns and prompts whenever free window drops below 10% (`auto_compact_when_low(..., min_free_ratio=0.10)`), and strictly clamps `system_prompt`, `user_prompt`, and `history` via `clamp_to_token_budget` so total input + output + multimodal vision tokens never exceed `preset.context_length`.
- **Two-Tier Memory** ([src/danibook/memory/tier1_desk.py](file:///Users/andrestobelem/ws/at/danibook/src/danibook/memory/tier1_desk.py), [src/danibook/memory/tier2_archive.py](file:///Users/andrestobelem/ws/at/danibook/src/danibook/memory/tier2_archive.py)):
  - `DeskMemory` scales and caps total prefix characters (`max_chars`) across all checked `Sources` dynamically based on `preset.context_length` so 36+ active books fit safely inside any model's context window (8K or 32K).
  - `ArchiveMemory` persists chunks in `data/cache/archive_fts5.sqlite` by `SHA-256` hash for sub-second startup and strips OCR control bytes (`ord(ch) < 32`) before prompt injection.
- **Material Design 3 UI & Bun Gateway** ([src/danibook/ui/notebooklm.html](file:///Users/andrestobelem/ws/at/danibook/src/danibook/ui/notebooklm.html), [src/bun/server.ts](file:///Users/andrestobelem/ws/at/danibook/src/bun/server.ts)):
  - Uses official `@material/web` components (`vendor/material-web.min.js`), native offline Mermaid (`vendor/mermaid.min.js`), a 3-way theme selector (`System` default, `Light`, `Dark`), animated thinking indicator, 5 curated `Collections`, `Sample` picker, notebook report export (Markdown/PDF), and 6 `Studio Artifacts` (`summary`, `mindmap`, `flashcards`, `glossary`, `timeline`, `quotes`).

## Execution Loop

1. **Ground**: Read target files, `GLOSSARY.md`, and relevant ADRs; confirm every symbol and constraint exists in the environment.
2. **Cut Surgically**: Modify only the lines required by the task, preserving canonical domain terms and existing docstrings.
3. **Prove**: Run `/opt/homebrew/bin/uv run pytest tests/` and verify live HTTP/UI behavior on `:7860` / `:7861`; every check must go green before reporting completion.
