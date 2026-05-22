import fitz
import chromadb
import ollama

from sentence_transformers import SentenceTransformer



def extract_text_from_pdf(pdf_path):

    doc = fitz.open(pdf_path)

    full_text = ""

    for page in doc:

        full_text += page.get_text()

    return full_text



def chunk_text(text, chunk_size=500, overlap=100):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks




client = chromadb.Client()

collection = client.create_collection(
    name="rag_collection"
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


pdf_path = "data/pdfs/manual.pdf"

text = extract_text_from_pdf(pdf_path)

chunks = chunk_text(text)

print(f"\nTotal chunks: {len(chunks)}")


embeddings = model.encode(chunks).tolist()

ids = [f"chunk_{i}" for i in range(len(chunks))]



collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=ids
)

print("\nChunks guardados en ChromaDB.")




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

print("\nCONTEXTO RECUPERADO:\n")
print(context)




prompt = f"""
Responde la pregunta usando SOLO la información
del contexto proporcionado.

Si la respuesta no está en el contexto,
di:
"No encontré esa información en el documento."

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