# Arquitectura Multimodal de Gemma 4

## 1. Visión y Proyección
Gemma 4 incorpora capacidades multimodales nativas que procesan tanto texto como imágenes. A diferencia de arquitecturas monolíticas de lenguaje puro, Gemma 4 utiliza un codificador visual basado en CLIP junto con un proyector no lineal (`mmproj`) que traduce las imágenes entrantes en una secuencia de tokens de embedding integrados directamente en el espacio latente del Transformer.

## 2. Ventana de Contexto y Gestión de Memoria
Con una ventana de atención de hasta 32.768 tokens y mecanismos de atención por ventanas deslizantes (Sliding Window Attention), el modelo es capaz de procesar documentos extensos y múltiples imágenes simultáneas dentro de un presupuesto de memoria acotado.
