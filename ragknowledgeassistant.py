import streamlit as st

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
