import chromadb
from sentence_transformers import SentenceTransformer

# Crear cliente Chroma
client = chromadb.Client()

# Crear colección
collection = client.create_collection(name="manual_collection")

# Modelo embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

# Textos de prueba
documents = [
    "El sensor mide temperatura",
    "El sensor térmico detecta calor",
    "Los motores hidráulicos generan presión"
]

# IDs únicos
ids = ["doc1", "doc2", "doc3"]

# Crear embeddings
embeddings = model.encode(documents).tolist()

# Guardar en ChromaDB
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=ids
)

print("\nEmbeddings guardados correctamente.")

# CONSULTA
query = "¿Cómo funciona un sensor de calor?"

query_embedding = model.encode([query]).tolist()

# Buscar similitud
results = collection.query(
    query_embeddings=query_embedding,
    n_results=2
)

print("\nRESULTADOS DE BÚSQUEDA:")
print(results)