import ollama

# Contexto recuperado (simulando retrieval)
context = """
El sensor térmico detecta cambios de calor
mediante un sistema de medición electrónica.
"""

# Pregunta usuario
question = "¿Cómo funciona el sensor térmico?"

# Prompt
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

# Llamar a Llama3
response = ollama.chat(
    model="llama3",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

# Mostrar respuesta
print("\nRESPUESTA DEL MODELO:\n")

print(response["message"]["content"])