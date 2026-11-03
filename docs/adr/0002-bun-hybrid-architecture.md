# Hybrid Bun Gateway and Metal Inference Daemon

- **Status**: accepted

Serving the Material 3 UI, static bundles, and SQLite FTS5 queries inside the same Python process that holds multi-gigabyte Gemma 4 GGUF weights and CLIP vision projectors in Apple Metal VRAM couples web restarts to expensive GPU reloads. We decouple the runtime into a **Bun Core Gateway** (`Bun.serve` + native `bun:sqlite` FTS5 + `@material/web` bundler on port 7861) and a dedicated **Metal Inference Daemon** (`llama-cpp-python` + `Gemma4ChatHandler` on port 7860). Web and search operations execute in sub-millisecond native code while GPU model lifecycle (loading, unloading, and on-demand Hugging Face downloads) remains isolated.
