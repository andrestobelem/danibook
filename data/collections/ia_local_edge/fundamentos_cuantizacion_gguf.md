# Fundamentos de Cuantización GGUF y Ejecución en Edge

## 1. ¿Qué es GGUF?
GGUF (GPT-Generated Unified Format) es el formato binario estándar introducido por el proyecto llama.cpp para empaquetar modelos de lenguaje grandes junto con sus metadatos (arquitectura de capas, tamaños de vocabulario, parámetros de contexto y configuración del proyector multimodal).

## 2. Tipos de Cuantización Comunes
* **Q4_K_M**: Método híbrido que cuantiza los pesos de atención a 4 bits utilizando matrices k-quants con precisión intermedia en las capas críticas. Ofrece el mejor balance entre consumo de VRAM y retención de perplejidad.
* **Q8_0**: Cuantización uniforme a 8 bits con mínima degradación perceptual, ideal para tareas de razonamiento formal y código.
* **F16**: Precisión flotante nativa utilizada principalmente en el proyector de visión CLIP (mmproj) para evitar distorsiones en los embeddings espaciales.

## 3. Aceleración en Apple Silicon (Metal Performance Shaders)
En arquitecturas Apple M-series, la memoria unificada (UMA) permite que tanto la CPU como la GPU accedan a los tensores sin necesidad de transferencias lentas por bus PCIe. Esto permite que modelos de 2B a 12B parámetros corran a velocidades de generación superiores a 40 tokens por segundo en hardware de consumo personal.
