import os
import re
import nltk
import json
import fitz  # PyMuPDF
from nltk.tokenize import sent_tokenize

# Download NLTK tokenizer
nltk.download("punkt_tab")

# Create output folders
os.makedirs("chunks", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Load PDF and extract text
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

pdf_path = "data/AI Training Document.pdf"  # <-- use your actual PDF name
raw_text = extract_text_from_pdf(pdf_path)

# Clean the extracted text
def clean_text(text):
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r" +", " ", text)
    return text.strip()

cleaned_text = clean_text(raw_text)

# Chunk the cleaned text into sentence-aware blocks
def chunk_text(text, min_words=100, max_words=300):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_word_count + word_count > max_words:
            if current_word_count >= min_words:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0
        current_chunk.append(sentence)
        current_word_count += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

chunks = chunk_text(cleaned_text)

# Save the chunks
with open("chunks/chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print(f"Total chunks created: {len(chunks)}")
