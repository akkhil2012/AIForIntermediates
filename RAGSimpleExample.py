
# Create fresh env
#python -m venv rag_env
#source rag_env/bin/activate       # Mac/Linux
#rag_env\Scripts\activate          # Windows

# Install everything clean
#pip install transformers torch sentence-transformers numpy huggingface_hub'''

from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Knowledge base
docs = [
    "Python is a high-level programming language.",
    "RAG stands for Retrieval-Augmented Generation.",
    "Paris is the capital of France.",
    "The sun is a star at the center of our solar system.",
]

# 2. Embed documents
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = model.encode(docs)

# 3. Ask a question
query = "What is RAG?"
query_embedding = model.encode([query])

# 4. Find most similar document
scores = np.dot(doc_embeddings, query_embedding.T).flatten()
best_match = docs[np.argmax(scores)]

print(f"Q: {query}")
print(f"A: {best_match}")




