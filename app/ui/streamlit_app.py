import os
import chromadb
from llm_service import generate_answer
from sentence_transformers import SentenceTransformer
import streamlit as st
from PIL import Image

from document_processor import process_document

# 1. Configuración inicial de la página
st.set_page_config(page_title="Chat PDF", layout="wide")

# Carpetas de datos y rutas
PDF_FOLDER = "data/pdfs"
CHROMA_PATH = "data/chroma_db"
os.makedirs(PDF_FOLDER, exist_ok=True)

# Inicializar st.session_state para controlar archivos ya procesados
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

# Encabezado con Logo y Título
try:
    logo = Image.open("app/assets/logo.png")
    col1, col2 = st.columns([1, 8])
    with col1:
        st.image(logo, width=100)
    with col2:
        st.markdown("# Enterprise Document AI")
except Exception:
    st.markdown("# Enterprise Document AI")

st.markdown("---")


@st.cache_resource
def load_resources():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        collection = client.get_collection("multimodal_rag")
    except Exception:
        collection = None

    model = SentenceTransformer("all-MiniLM-L6-v2")

    return collection, model


collection, model = load_resources()

# 3. Subida y procesamiento de archivos PDF
uploaded_file = st.file_uploader("📄 Upload Enterprise Document", type=["pdf"])

if uploaded_file is not None:
    if st.session_state.processed_file != uploaded_file.name:

        save_path = os.path.join(PDF_FOLDER, uploaded_file.name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Analyzing document and building the knowledge base..."):
            result = process_document(save_path)

        # Limpiar la caché de Streamlit y recargar Chroma
        st.cache_resource.clear()
        collection, model = load_resources()

        # Guardar en el estado
        st.session_state.processed_file = uploaded_file.name
        st.success(f"Document processed ({result.get('chunks', 0)} chunks)")

st.markdown("---")

processing = uploaded_file is not None and (
    st.session_state.processed_file != uploaded_file.name
)

# 4. Sección del Chat / Consulta
if st.session_state.processed_file is not None:

    question = st.text_area(
        "", placeholder="Ask anything about your document...", disabled=processing
    )

    search = st.button("Ask AI", disabled=processing)

    if search and question:
        if collection is None:
            st.error("No se encontró la colección en la base de datos Chroma.")
        else:
            # A. Búsqueda Vectorial
            query = model.encode([question]).tolist()
            
            # Asignamos el resultado a 'results'
            results = collection.query(
                query_embeddings=query,
                n_results=8
            )

            if results.get("documents") and results["documents"][0]:
                # Unir los fragmentos de contexto recuperados
                context = "\n\n".join(results["documents"][0])
                metadata = results["metadatas"][0][0]

                # B. Generación de Respuesta
                prompt = f"""
Eres un asistente especializado en responder preguntas sobre documentos PDF.

REGLAS:

- Responde únicamente utilizando la información del CONTEXTO.
- Si la respuesta aparece en varios fragmentos, combínalos.
- No inventes información.
- Si la información no aparece en el contexto responde exactamente:

"No encontré esa información en el documento."

- Si la pregunta hace referencia a un personaje, evento, capítulo o concepto, busca toda la información disponible en el contexto antes de responder.

====================
CONTEXTO
====================

{context}

====================
PREGUNTA
====================

{question}

====================
RESPUESTA
====================
"""
                answer = generate_answer(prompt)

                # C. Mostrar Respuestas y Referencias
                st.subheader("Respuesta")
                st.write(answer)

                st.subheader("Referencia")
                st.write(f"Página {metadata.get('page', 'Desconocida')}")

                # D. Procesamiento de imágenes
                show_intent = any(
                    x in question.lower() for x in ["imagen", "mostrar", "muestra"]
                )

                images = []
                if show_intent:
                    current_page = int(metadata.get("page", 0))

                    # Extraer imágenes del resultado principal
                    for item in results["metadatas"][0]:
                        if item.get("images") and item["images"].strip():
                            images.extend(item["images"].split("|"))

                    # Buscar en páginas adyacentes si no hay en el resultado directo
                    if not images:
                        nearby = collection.get()
                        for meta in nearby.get("metadatas", []):
                            try:
                                page_num = int(meta.get("page", -999))
                                if abs(page_num - current_page) <= 5 and meta.get("images"):
                                    images.extend(meta["images"].split("|"))
                            except (ValueError, TypeError):
                                pass

                    # Eliminar duplicados y comprobar existencia en disco
                    images = list(
                        dict.fromkeys(
                            [
                                img.strip()
                                for img in images
                                if img.strip() and os.path.exists(img.strip())
                            ]
                        )
                    )

                    st.subheader("🎨🖌 Imágenes Relacionadas")

                    if images:
                        st.session_state.current_image = 0

                        col1, col2, col3 = st.columns([2, 4, 2])
                        with col2:
                            st.image(
                                images[st.session_state.current_image],
                                width=150,
                            )
                    else:
                        st.warning("No se encontraron imágenes para esta sección del PDF.")
            else:
                st.warning("No se encontró información relevante en el documento.")

else:
    st.info("📄 Upload a PDF document to start chatting.")