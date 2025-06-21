import os
import json
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# Load chunked data
with open("chunks/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Generate embeddings
print("🔄 Generating embeddings...")
embeddings = model.encode(chunks, show_progress_bar=True)

# Convert to FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save index and chunk mapping
os.makedirs("vectordb", exist_ok=True)

faiss.write_index(index, "vectordb/index.faiss")

# Save metadata to map chunk index to text
with open("vectordb/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print(f"Vector DB created and saved with {len(chunks)} chunks.")
