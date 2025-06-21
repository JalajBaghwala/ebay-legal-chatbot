from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from src.generator import load_llm
from src.retriever import load_vector_retriever

def build_rag_pipeline():
    llm = load_llm()
    retriever = load_vector_retriever()

    # Optional: Custom prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant. Use the following context to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain
