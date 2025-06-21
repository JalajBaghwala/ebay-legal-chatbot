from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

def load_llm():
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")

    )
