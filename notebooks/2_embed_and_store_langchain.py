import os
import json
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load chunks
with open("chunks/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Use LangChain-compatible embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS vectorstore
vectorstore = FAISS.from_texts(chunks, embedding_model)

# Save using LangChain’s saving method
os.makedirs("vectordb", exist_ok=True)
vectorstore.save_local("vectordb")
 
print(f"Vector DB saved to vectordb/ using LangChain format.")
