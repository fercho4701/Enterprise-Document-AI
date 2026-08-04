# Enterprise Document AI

An AI-powered document assistant capable of processing enterprise PDF documents, extracting text and images, building a semantic knowledge base, and answering questions using Retrieval-Augmented Generation (RAG).

---

## Features

- 📄 Upload enterprise PDF documents
- 🤖 Ask questions in natural language
- 🧠 Semantic search using Sentence Transformers
- ⚡ Answer generation powered by Groq LLM
- 🗄 Vector database with ChromaDB
- 🖼 Automatic image extraction from PDFs
- 📚 Context-aware Retrieval-Augmented Generation (RAG)
- 🌐 Interactive Streamlit interface

---

## Architecture

```text
PDF
 │
 ▼
Text & Image Extraction
 │
 ▼
Chunking
 │
 ▼
Sentence Transformers
 │
 ▼
ChromaDB
 │
 ▼
Semantic Retrieval
 │
 ▼
Groq LLM
 │
 ▼
Answer
```

---

## Technologies

### Backend

- Python
- Groq API
- ChromaDB
- Sentence Transformers

### PDF Processing

- PyMuPDF (fitz)

### Frontend

- Streamlit

### AI Model

- Llama 3.3 70B Versatile (Groq)

---

## Project Structure

```text
app/
│
├── assets/
│
└── ui/
    ├── streamlit_app.py
    ├── document_processor.py
    └── llm_service.py

data/
│
├── chroma_db/
├── images/
└── pdfs/

requirements.txt
.env.example
README.md
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/fercho4701/IA-PDF.git
```

### Create virtual environment

```bash
python -m venv venv
```

### Activate environment

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment Variables

Create a `.env` file based on `.env.example`

```text
GROQ_API_KEY=your_api_key_here
```

---

## Run the application

```bash
streamlit run app/ui/streamlit_app.py
```

---

## Example Questions

- What is this document about?
- Summarize Chapter 3.
- Who is Pedro Pablo Gaviota?
- Show me the image related to this section.
- What does the document say about artificial intelligence?

---

## Current Capabilities

- Semantic document search
- Context-aware question answering
- Automatic PDF processing
- Image extraction
- Interactive chat interface
- Enterprise document analysis

---

## Roadmap

- Hybrid Search
- Cross-Encoder Re-ranking
- Multi-document support
- Conversation memory
- Cloud deployment
- Source citations
- OCR support

---

## Author

**Luis Fernando González Molina**

Systems Engineer

GitHub:
https://github.com/fercho4701

---

## License

This project is licensed under the MIT License.
