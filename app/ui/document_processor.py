import fitz
import chromadb
import os
from sentence_transformers import SentenceTransformer

IMAGE_FOLDER = "data/images"
CHROMA_PATH = "data/chroma_db"

os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)


# ==========================
# EXTRAER TEXTO + IMÁGENES
# ==========================

def extract_multimodal_data(pdf_path):

    doc = fitz.open(pdf_path)

    pages_data = []

    for page_number, page in enumerate(doc):

        text = page.get_text()

        image_list = page.get_images(full=True)

        saved_images = []

        for img_index, img in enumerate(image_list):

            xref = img[0]

            base_image = doc.extract_image(xref)

            image_bytes = base_image["image"]

            image_ext = base_image["ext"]

            image_name = (
                f"page_{page_number+1}_img_{img_index}.{image_ext}"
            )

            image_path = os.path.join(
                IMAGE_FOLDER,
                image_name
            )

            with open(
                image_path,
                "wb"
            ) as f:

                f.write(
                    image_bytes
                )

            saved_images.append(
                image_path
            )

        pages_data.append({

            "page": page_number + 1,

            "text": text,

            "images": saved_images

        })

    return pages_data


# ==========================
# CHUNKING
# ==========================

def chunk_text(
    text,
    chunk_size=1000,
    overlap=200
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(
            text[start:end]
        )

        start += (
            chunk_size
            -
            overlap
        )

    return chunks


# ==========================
# CREAR CHROMA PERSISTENTE
# ==========================

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

def process_document(pdf_path):

    try:
        client.delete_collection(
            "multimodal_rag"
        )
    except:
        pass

    collection = client.create_collection(
        name="multimodal_rag"
    )

    pages_data = extract_multimodal_data(pdf_path)

    all_docs = []
    all_embeddings = []
    all_ids = []
    all_metadata = []

    counter = 0

    for page in pages_data:

        chunks = chunk_text(page["text"])

        for chunk in chunks:

            embedding = model.encode(chunk).tolist()

            all_docs.append(chunk)
            all_embeddings.append(embedding)
            all_ids.append(f"chunk_{counter}")

            all_metadata.append({
                "page": str(page["page"]),
                "document": pdf_path,
                "images": "|".join(page["images"])
            })

            counter += 1

    collection.add(
        documents=all_docs,
        embeddings=all_embeddings,
        ids=all_ids,
        metadatas=all_metadata
    )

    return {
        "status": "success",
        "chunks": counter
    }