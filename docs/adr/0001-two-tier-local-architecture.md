# Local Two-Tier Memory Architecture

- **Status**: accepted

Daniel Kusnir requires an offline, privacy-first book-drafting environment on Apple Silicon and Windows where full chapters stay in narrative focus while hundreds of reference PDFs remain searchable without full-prompt re-evaluation latency. We split Notebook memory into **Desk (Tier 1)**—an ordered, budget-capped prefix of active Sources for continuous context—and **Archive (Tier 2)**—an incremental SQLite FTS5 BM25 index returning cited fragments on demand. This eliminates RAG prefill bottlenecks on active chapters while keeping large reference libraries within the 32K context window of quantized local Gemma 4 models.
