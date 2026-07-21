"""
query.py
--------
Loads the ChromaDB vector database and retrieves the most relevant
documents for a user query.
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    CHROMA_DB_DIR,
    EMBEDDING_MODEL,
    TOP_K,
)

# ==========================================================
# Load Embedding Model
# ==========================================================

print("Loading embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

# ==========================================================
# Load ChromaDB
# ==========================================================

print("Loading ChromaDB...")

vector_db = Chroma(
    persist_directory=str(CHROMA_DB_DIR),
    embedding_function=embedding_model,
)

# ==========================================================
# Retrieve Documents
# ==========================================================

def retrieve_documents(question: str):

    results = vector_db.similarity_search(
        question,
        k=TOP_K
    )

    context = ""
    sources = []

    for doc in results:

        context += doc.page_content + "\n\n"

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")

        if page is not None:
            sources.append(f"{source} (Page {page})")
        else:
            sources.append(source)

    return context, sources