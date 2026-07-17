from sentence_transformers import SentenceTransformer

# Cargar modelo embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

# Texto de prueba
sentences = [
    "El sensor mide temperatura",
    "El sensor térmico detecta calor",
    "Los motores hidráulicos generan presión"
]

# Crear embeddings
embeddings = model.encode(sentences)

# Mostrar resultados
for i, embedding in enumerate(embeddings): 
    print("\n" + "=" * 50)
    print(f"TEXTO: {sentences[i]}")
    print(f"VECTOR (primeros 10 valores):")

    print(embedding[:10])