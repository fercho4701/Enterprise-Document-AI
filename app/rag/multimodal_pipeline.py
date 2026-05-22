import fitz
import chromadb
import ollama
import os

from sentence_transformers import SentenceTransformer



PDF_PATH = "data/pdfs/manual.pdf"

IMAGE_FOLDER = "data/images"

os.makedirs(IMAGE_FOLDER, exist_ok=True)



def extract_multimodal_data(pdf_path):

    doc = fitz.open(pdf_path)

    pages_data = []

    for page_number, page in enumerate(doc):

        print(f"Procesando página {page_number + 1}")

       
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

            with open(image_path, "wb") as f:

                f.write(image_bytes)

            saved_images.append(image_path)

        page_info = {
            "page": page_number + 1,
            "text": text,
            "images": saved_images
        }

        pages_data.append(page_info)

    return pages_data




def chunk_text(text, chunk_size=500, overlap=100):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks



pages_data = extract_multimodal_data(PDF_PATH)



client = chromadb.Client()

collection = client.create_collection(
    name="multimodal_rag"
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


all_chunks = []
all_embeddings = []
all_ids = []
all_metadatas = []

chunk_counter = 0

for page_data in pages_data:

    chunks = chunk_text(page_data["text"])

    for chunk in chunks:

        embedding = model.encode(chunk).tolist()

        chunk_id = f"chunk_{chunk_counter}"

        metadata = {
            "page": page_data["page"],
            "document": PDF_PATH,
            "images": ", ".join(page_data["images"])
        }

        all_chunks.append(chunk)

        all_embeddings.append(embedding)

        all_ids.append(chunk_id)

        all_metadatas.append(metadata)

        chunk_counter += 1


collection.add(
    documents=all_chunks,
    embeddings=all_embeddings,
    ids=all_ids,
    metadatas=all_metadatas
)

print("\nChunks multimodales guardados.")



question = input("\nHaz una pregunta: ")



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



print("\nMETADATA RECUPERADA:\n")

for item in metadata:

    print(item)


prompt = f"""
Responde la pregunta usando SOLO la información
del contexto proporcionado.

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

print("\nRESPUESTA FINAL:\n")

print(response["message"]["content"])