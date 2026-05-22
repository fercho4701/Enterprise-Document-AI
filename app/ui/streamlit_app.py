import streamlit as st
import fitz
import chromadb
import ollama
import os

from sentence_transformers import SentenceTransformer


PDF_PATH = "data/pdfs/manual.pdf"

IMAGE_FOLDER = "data/images"




st.set_page_config(
    page_title="Multimodal RAG",
    layout="wide"
)

st.title("📚 Multimodal RAG Chat")

st.write(
    "Haz preguntas sobre el documento PDF"
)



client = chromadb.PersistentClient(
    path="data/chroma_db"
)

collection = client.get_or_create_collection(
    name="multimodal_rag"
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)



question = st.text_input(
    "Escribe tu pregunta:"
)




if question:

   
    query_embedding = model.encode(
        [question]
    ).tolist()

    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=2
    )

    context = "\n".join(
        results["documents"][0]
    )

    metadata = results["metadatas"][0]

   
    prompt = f"""
    Responde usando SOLO el contexto.

    CONTEXTO:
    {context}

    PREGUNTA:
    {question}

    RESPUESTA:
    """

    response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]



    st.subheader("🧠 Respuesta")

    st.write(answer)

   

    st.subheader("📄 Referencias")

    for item in metadata:

        st.write(f"Página: {item['page']}")

        st.write(f"Documento: {item['document']}")

        # MOSTRAR IMÁGENES
        if item["images"]:

            image_paths = item["images"].split(",")

            for image_path in image_paths:

                image_path = image_path.strip()

                if os.path.exists(image_path):

                    st.image(
                        image_path,
                        width=400
                    )