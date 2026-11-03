# Investigación: Arquitectura Local tipo NotebookLM con Gemma 4 & Hugging Face (`danibook`)

**Fecha**: 2026-11-03  
**Contexto**: Diseño de una alternativa local, rápida e incremental a NotebookLM para **Daniel Kusnir** (escritura de su libro).

---

## 1. Diagnóstico: ¿Por qué le resultó tan lento su intento anterior con RAG?

Cuando se monta un RAG local tradicional sobre una máquina de escritorio/laptop, la lentitud suele venir de **tres cuellos de botella acumulativos**:

1. **Re-indexación o Embedding pesado**: Convertir documentos enteros en fragmentos (chunks), calcular embeddings y cruzarlos cada vez que cambia un archivo.
2. **El costo de "Prefill" (Time-To-First-Token)**: En un RAG mal configurado (o sin caché de contexto), cada vez que el usuario hace una pregunta, el motor local vuelve a "leer" desde cero los fragmentos recuperados (3.000 a 15.000 tokens). En hardware local, procesar la entrada (prefill) tarda varios segundos o minutos antes de escupir la primera palabra.
3. **Pérdida de visión narrativa (El problema de los libros)**: Un libro requiere coherencia global (hilos conductores, tono, capítulos previos). El RAG tradicional corta el texto en pedazos aislados de 500 palabras, perdiendo la estructura de la obra y obligando al usuario a repreguntar.

---

## 2. Por qué **Gemma 4** cambia las reglas del juego en local

Google lanzó la familia **Gemma 4** (licencia Apache 2.0) con innovaciones arquitectónicas pensadas exactamente para este escenario:

### Variantes ideales para correr local (Livianas y Rápidas)
| Modelo | Parámetros Efectivos | Ventana de Contexto | Casos de uso / Hardware |
| :--- | :--- | :--- | :--- |
| **`google/gemma-4-E2B-it`** | ~2.3B (con PLE*) | **128K tokens** (~95.000 palabras) | Ultra rápido, corre hasta en CPU o laptops modestas. |
| **`google/gemma-4-E4B-it`** | ~4.5B (con PLE*) | **128K tokens** (~95.000 palabras) | **El punto dulce (Sweet Spot)** para laptops de 8GB–16GB RAM/VRAM. |
| **`google/gemma-4-12B-it`** | 12B Dense | **256K tokens** (~190.000 palabras) | Ideal si tiene 16GB–24GB+ de RAM unificada (Mac M-series) o buena GPU. |
| **`google/gemma-4-26B-A4B-it`** | **3.8B activos** (MoE) | **256K tokens** | Velocidad de un modelo de 4B con la inteligencia de uno de 26B (requiere ~18-24GB RAM para alojar los pesos). |

*\*PLE (Per-Layer Embeddings): Técnica de Gemma 4 que le da a los modelos chicos (E2B/E4B) una capacidad semántica mucho mayor sin penalizar la velocidad de cómputo.*

### La ventaja crítica: Atención Híbrida + Compresión de KV Cache
En modelos viejos, meter 50.000 palabras en el contexto hacía explotar la memoria RAM y frenaba cada respuesta. **Gemma 4** usa una **arquitectura híbrida 5:1** (5 capas de ventana deslizante local por cada 1 capa global) y reutilización de Keys/Values, reduciendo el consumo de memoria de contexto en casi un **40%**.

---

## 3. El Secreto para que "Agregar poco a poco" sea Instantáneo: **Incremental Prefix / KV Caching**

Daniel quiere **ir agregando documentos/capítulos poco a poco sin que tarde tanto**. En lugar de usar únicamente un RAG lento, podemos combinar el ecosistema de **Hugging Face** para hacer **Ingesta Incremental con KV Cache Persistente**:

### ¿Cómo funciona?
1. **Estructura tipo "Append-Only" (Apendizar al final)**:
   - Los modelos procesan el texto de izquierda a derecha.
   - Si el Documento 1 y el Documento 2 ya fueron leídos por Gemma 4, su estado mental matemático (el **KV Cache**) se guarda en disco/memoria.
   - Cuando Daniel agrega el **Documento 3** (un nuevo capítulo o nota de investigación), el modelo **NO vuelve a procesar el Documento 1 ni el 2**. Carga el estado ya calculado desde el disco y **solo procesa los tokens nuevos del Documento 3** (tarda apenas un par de segundos).
2. **Preguntas instantáneas**:
   - Cuando Daniel hace una pregunta en el chat, el modelo ya tiene todo el libro/documentos pre-digeridos en su caché. El *Time-To-First-Token* (tiempo hasta que empieza a responder) baja de 40 segundos a **menos de 1 segundo**.

---

## 4. Herramientas de Hugging Face para Construirlo

Dentro del ecosistema de Hugging Face tenemos dos caminos técnicos principales (o una combinación según el sistema operativo de Daniel):

### A. Motor de Inferencia y Caché de Estado
1. **`transformers` (`StaticCache` / `HybridCache` + `bitsandbytes` / `TorchAO`)**:
   - Permite controlar de forma nativa la caché de tensores (`past_key_values`), serializarla a disco con `torch.save()` / `safetensors` cada vez que Daniel agrega un documento, y reanudarla al instante.
2. **`mlx-lm` / `mlx-vlm` (Si Daniel usa Mac con Apple Silicon M1/M2/M3/M4)**:
   - Integrado directamente con el Hub de Hugging Face (`mlx-community/gemma-4-E4B-it-4bit`). Tiene soporte nativo para guardar archivos `.safetensors` de Prompt Cache en disco y hacer prefill incremental ultrarrápido.
3. **`llama-cpp-python` + `huggingface_hub` (Si usa Windows / Linux con CPU o GPU NVIDIA)**:
   - Descarga los modelos GGUF directo desde Hugging Face y usa `LlamaDiskCache` / sesiones persistentes para guardar el estado exacto de los documentos ya leídos.

### B. Procesamiento de Documentos e Búsqueda Híbrida (Para cuando el corpus supere los 128K/256K tokens)
Si en algún momento suma decenas de libros de referencia además de sus borradores:
- **`Docling` / `MarkItDown`** (en el ecosistema HF): Para convertir sus PDFs, Word (`.docx`) o notas a Markdown limpio con citas exactas de página/párrafo.
- **`sentence-transformers` + `chonkie` / `LanceDB` (o `sqlite-vec`)**: Un índice ultraligero en un único archivo local. Al agregar un documento, solo se indexa el archivo nuevo (incremental en milisegundos).
- **`smolagents` o `Gradio`**: Para darle una interfaz limpia estilo NotebookLM donde a la izquierda vea sus documentos/capítulos, al centro su borrador o guía, y a la derecha el chat con citas clicables.

---

## 5. Preguntas Clave para Definir la Arquitectura Exacta

Para elegir el stack exacto sin sobre-ingeniería:
1. **¿Qué computadora usa Daniel?** (¿Mac con chip M1/M2/M3/M4, PC con Windows/Linux, cuánta RAM tiene y si tiene placa de video NVIDIA?).
2. **¿Qué formato tienen los archivos que va agregando?** (¿Archivos Word `.docx`, PDFs de investigación, archivos de texto/Markdown, notas de audio?).
3. **¿Qué volumen aproximado maneja?** (¿Solo los capítulos de su libro —que entran enteros en la ventana de 128K de Gemma 4 E4B sin siquiera necesitar RAG—, o también decenas de PDFs pesados de bibliografía?).
