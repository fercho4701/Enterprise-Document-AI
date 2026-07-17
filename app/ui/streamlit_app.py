import streamlit as st
import chromadb
import ollama
import os

from sentence_transformers import SentenceTransformer


CHROMA_PATH = "data/chroma_db"


st.set_page_config(
    page_title="Chat PDF",
    layout="wide"
)

st.title(
    "📚🤖 Chat PDF"
)


client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    "multimodal_rag"
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


question = st.text_input(
    "Pregunta:"
)


if question:

    query = model.encode(
        [question]
    ).tolist()

    results = collection.query(

        query_embeddings=query,

        n_results=1
    )

    context = results[
        "documents"
    ][0][0]

    metadata = results[
        "metadatas"
    ][0][0]


    prompt = f"""
Responde SOLO usando el contexto.

CONTEXTO:
{context}

PREGUNTA:
{question}
"""


    response = ollama.chat(

        model="llama3",

        messages=[

            {

                "role":"user",

                "content":prompt

            }

        ]

    )


    answer = response[
        "message"
    ][
        "content"
    ]


    st.subheader(
        "Respuesta"
    )

    st.write(
        answer
    )


    st.subheader(
        "Referencia"
    )

    st.write(
        f"Página {metadata['page']}"
    )


    show = any(

        x in question.lower()

        for x in [

            "imagen",

            "mostrar",

            "muestra"

        ]

    )


    if show:

     images = []

    current_page = int(
        metadata["page"]
    )

    for item in results["metadatas"][0]:

        if (
            item["images"]
            and
            item["images"].strip()
        ):

            images.extend(
                item["images"].split("|")
            )


    if not images:

        # buscar imágenes cercanas

        for page_offset in range(
            1,
            6
        ):

            page_before = (
                current_page
                -
                page_offset
            )

            page_after = (
                current_page
                +
                page_offset
            )

            nearby = collection.get()

            for meta in nearby[
                "metadatas"
            ]:

                try:

                    page_num = int(
                        meta["page"]
                    )

                    if (
                        page_num
                        in [
                            page_before,
                            page_after
                        ]
                        and
                        meta["images"]
                    ):

                        images.extend(
                            meta[
                                "images"
                            ].split("|")
                        )

                except:
                    pass


    st.subheader(
        "🖼️ Imágenes"
    )

    shown = set()

    for img in images:

        img = img.strip()

        if (
            img
            and
            os.path.exists(
                img
            )
            and
            img not in shown
        ):

            st.image(
                img,
                width=150
            )

            shown.add(
                img
            )


    if not shown:

        st.warning(
            "No se encontraron imágenes para esta sección del PDF."
        )

        st.subheader(
            "Imágenes"
        )


        for img in images:

            if (

                img
                and
                os.path.exists(
                    img
                )

            ):

                st.image(
                    img,
                    width=700
                )