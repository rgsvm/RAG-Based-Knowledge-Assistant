import streamlit as st
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    return text

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks

import openai

def get_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

import faiss
import numpy as np

def build_vector_store(chunks):
    embeddings = [get_embedding(chunk) for chunk in chunks]

    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings).astype("float32"))

    return index, embeddings, chunks

def search(query, index, chunks, embeddings, k=3):
    query_embedding = get_embedding(query)

    distances, indices = index.search(
        np.array([query_embedding]).astype("float32"),
        k
    )

    results = [chunks[i] for i in indices[0]]
    return results

def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a helpful assistant.
Answer the question based only on the context below.

Context:
{context}

Question:
{query}

Answer clearly and concisely:
"""

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

st.title("RAG Knowledge Assistant")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    st.write(text[:1000])

st.title("📚 RAG Knowledge Assistant")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(text)

    index, embeddings, chunks = build_vector_store(chunks)

    st.success("Document processed!")

    query = st.text_input("Ask a question")

    if query:
        relevant_chunks = search(query, index, chunks, embeddings)
        answer = generate_answer(query, relevant_chunks)

        st.write("### Answer")
        st.write(answer)
