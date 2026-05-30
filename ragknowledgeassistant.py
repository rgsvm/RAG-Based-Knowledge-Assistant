import streamlit as st

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
