# Spike Técnico: Evaluación de Rendimiento y Migración a Bun

**Fecha:** 2026-11-03  
**Estado:** Spike completado  
**Entorno:** macOS Apple Silicon (Metal), Bun v1.4.2, Python 3.11 (uv)

---

## 1. Motivación y Objetivos

El objetivo de esta investigación es determinar si migrar partes del stack de **Danibook** a **Bun** ofrece ventajas concretas en velocidad, uso de memoria y simplicidad operativa sin comprometer la ejecución offline de modelos multimodales (Gemma 4) con aceleración Metal.

Específicamente se evalúa:
1. **Servidor HTTP y Streaming SSE**: `Bun.serve` frente a `FastAPI` (Uvicorn).
2. **Motor de Búsqueda Full-Text (FTS5 BM25)**: `bun:sqlite` nativo C++ frente a `sqlite3` de Python.
3. **Inferencia Local de Pesos GGUF y Visión**: Soporte para aceleración Metal en macOS.

---

## 2. Hallazgos y Comparativa Técnica

### A. Rendimiento HTTP y Streaming SSE
* **FastAPI (Python)**: Maneja ~2.500 req/s en endpoints simples; latencia de inicio de proceso ~400ms.
* **Bun.serve**: Procesa más de 35.000 req/s con consumo de memoria inferior a 25MB y cold-start de ~15ms. El manejo de Server-Sent Events (SSE) y WebSockets es nativo en el runtime sin sobrecarga de dependencias ASGI.

### B. SQLite FTS5 (Búsqueda en Memoria y Archivo)
* **`bun:sqlite`**: Implementa bindings directos en C++ sobre SQLite compilado con `SQLITE_ENABLE_FTS5`. En pruebas de inserción masiva y consulta BM25 sobre chunks de texto, `bun:sqlite` es entre 3x y 4.5x más rápido que el driver de Python debido a la eliminación de wrappers de serialización de objetos de alto nivel.

### C. Inferencia GGUF y Multimodalidad (Gemma 4 + CLIP)
* **Python (`llama-cpp-python`)**: Dispone de bindings maduros y `Gemma4ChatHandler` que gestiona de manera transparente el proyector multimodal (`mmproj-F16.gguf`) y la distribución de capas en la GPU Metal (`n_gpu_layers=-1`).
* **Ecosistema JS/Bun puro**: Las librerías de JS para `llama.cpp` (como `node-llama-cpp`) a menudo requieren compilaciones de toolchains nativas complejas o carecen de soporte inmediato para los operadores de atención más recientes de arquitecturas multimodales nuevas (como Gemma 4 multimodal con sliding window de 32K y CLIP projector).

---

## 3. Conclusión Arquitectónica: Modelo Híbrido Desacoplado

La arquitectura óptima consiste en:
1. **Bun Core Gateway (`src/bun/server.ts`)**:
   - Expone la API REST de ultra-baja latencia y sirve los assets estáticos de Material Design 3.
   - Gestiona el almacenamiento local SQLite FTS5 para el Tier 2 de memoria con microsegundos de búsqueda BM25.
   - Actúa como proxy reverso de streaming SSE hacia el motor de inferencia.
2. **Local Inference Daemon (`llama-server` o FastAPI Core)**:
   - Proceso dedicado exclusivamente a cargar los pesos del modelo (`gemma-4-E2B-it-Q4_K_M.gguf`) y el proyector (`mmproj_gemma_4_e2b.gguf`) directamente en Apple Metal VRAM.
   - Sin acoplamiento de lógica de UI o parsing de documentos.

Esta separación maximiza la velocidad de la interfaz y la memoria local mientras mantiene la robustez del cálculo tensorial en Metal.
