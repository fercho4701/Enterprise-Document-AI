def chunk_text(text, chunk_size=500, overlap=100):
    """
    Divide el texto en fragmentos más pequeños.
    """

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# PRUEBA
if __name__ == "__main__":

    sample_text = """
    La teoría general de sistemas busca comprender
    los principios aplicables a cualquier sistema.
    """

    chunks = chunk_text(sample_text)

    for i, chunk in enumerate(chunks):
        print(f"\nCHUNK {i+1}")
        print(chunk)