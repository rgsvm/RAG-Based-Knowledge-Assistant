Built a Retrieval-Augmented Generation (RAG) knowledge assistant using embeddings and 
vector search to enable context-aware question answering over custom documents, 
integrating OpenAI LLMs with FAISS-based semantic retrieval and a Streamlit interface.

Architecture:
User Question
      ↓
Query Embedding
      ↓
Vector DB (FAISS)
      ↓
Top-K Relevant Chunks
      ↓
LLM (OpenAI)
      ↓
Final Answer

Tech:
Python
Streamlit (UI)
OpenAI API (LLM + embeddings)
FAISS (vector database)
PyPDF2 / pdfminer (PDF reading)
LangChain optional
