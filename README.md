# Multimodal RAG System

Sistema RAG (Retrieval-Augmented Generation) multimodal capaz de procesar documentos PDF técnicos, extraer texto e imágenes, y responder preguntas utilizando contexto semántico y visual.

---

# Características

- Procesamiento de PDFs técnicos
- Extracción de texto e imágenes
- Chunking inteligente
- Embeddings semánticos
- Vector Database con ChromaDB
- Retrieval contextual
- Generación de respuestas con Llama3 (Ollama)
- Asociación texto ↔ imagen
- Interfaz interactiva con Streamlit

---

# Arquitectura

PDF → Extracción → Chunking → Embeddings → ChromaDB → Retrieval → LLM → Respuesta

---

# Tecnologías utilizadas

## Backend
- Python
- Ollama
- Llama3
- ChromaDB
- SentenceTransformers

## Procesamiento PDF
- PyMuPDF (fitz)

## Frontend
- Streamlit

---

# Estructura del proyecto

```bash
app/
│
├── ingestion/
├── rag/
├── ui/
│
data/
│
├── chroma_db/
├── images/
├── pdfs/
```

# Instalación

## 1. Clonar repositorio

```bash
git clone URL_DEL_REPOSITORIO
```

## 2. Crear entorno virtual

```bash
python -m venv venv
```

## 3. Activar entorno virtual

### Windows

```bash
.\venv\Scripts\Activate
```

## 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

# Instalar Ollama

Descargar desde:

https://ollama.com/download

---

# Descargar modelo

```bash
ollama run llama3
```

---

# Ejecutar pipeline multimodal

```bash
python app/rag/multimodal_pipeline.py
```

---

# Ejecutar interfaz

```bash
streamlit run app/ui/streamlit_app.py
```

---

# Ejemplo de preguntas

- ¿Cómo funciona el sensor térmico?
- ¿Qué dice el documento sobre teoría de sistemas?
- ¿Qué muestra el diagrama principal?

---

# Características multimodales

El sistema:
- Extrae imágenes desde PDFs
- Relaciona texto con imágenes cercanas
- Devuelve metadata:
  - documento
  - página
  - imágenes asociadas

---

# Futuras mejoras

- OCR avanzado
- Soporte multi-documento
- Historial conversacional
- Deploy cloud
- Mejor retrieval visual

---

# Autor

Luis Fernando González Molina