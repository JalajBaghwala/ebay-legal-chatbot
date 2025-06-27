import streamlit as st
from src.pipeline import build_rag_pipeline
from streamlit_chat import message  # optional for advanced chat formatting
from PIL import Image
from langchain_openai import ChatOpenAI
import time

# Page config
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# Sidebar customization
with st.sidebar:
    # st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=180)
    st.image("https://upload.wikimedia.org/wikipedia/commons/1/1b/EBay_logo.svg", width=180)
    st.title("📄 eBay Legal Assistant Chatbot")
    st.markdown("""
    This chatbot lets you ask questions about your document using a RAG pipeline.

    - Uses FAISS for retrieval
    - GPT-3.5 for answers
    - Streamlit for UI
    """)
    st.markdown("---")
    st.markdown("Built by Jalaj Baghwala")

# Title and header
st.markdown("""
    <h1 style='text-align: center;'>💬 Ask Your eBay Legal Assistant</h1>
    <h4 style='text-align: center; color: gray;'>A RAG-based Q&A assistant powered by GPT-3.5</h4>
    <br>
""", unsafe_allow_html=True)

# Initialize RAG pipeline
qa_chain = build_rag_pipeline()

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input box
user_input = st.chat_input("Type your question here...")
if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # GPT response simulation with spinner
    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            response = qa_chain.invoke({"query": user_input})
            answer = response["result"]
            sources = response.get("source_documents", [])


            if not sources:
                # No relevant documents found – fallback to pure GPT
                llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
                fallback_answer = llm.invoke(user_input)
                answer = fallback_answer.content
            else:
                answer = response["result"]
            
            st.markdown(answer)

                
            if sources:
                with st.expander("📚 Show Retrieved Contexts"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Chunk {i+1}:**\n```\n{doc.page_content[:1000]}\n```")


    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Optional enhancement: Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        🧠 Powered by FAISS + GPT-3.5 + Streamlit
    </div>
""", unsafe_allow_html=True)
