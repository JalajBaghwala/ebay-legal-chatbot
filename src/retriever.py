# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings



def load_vector_retriever(db_path="vectordb"):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # ✅ This will load FAISS index + metadata correctly
    vectorstore = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
    
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
