"""
ingest.py
---------
Reads PDF and TXT files, crawls configured websites, splits everything
into chunks, generates embeddings, and builds the ChromaDB vector database.

Run:
    python ingest.py              # PDFs + TXTs + website crawl
    python ingest.py --no-crawl   # PDFs + TXTs only
    python ingest.py --only-crawl # crawl only
"""

import sys
import re
import shutil

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    PDF_DIR,
    TEXT_DIR,
    CHROMA_DB_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from crawler import crawl_websites


# ==========================================================
# Clean Extracted Text
# ==========================================================

def clean_text(text: str) -> str:
    """
    Clean PDF extracted text before embedding.
    """

    # Remove multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove multiple blank lines
    text = re.sub(r"\n{2,}", "\n\n", text)

    # Join words where every character is separated by spaces.
    # Example:
    # S o f t w a r e -> Software
    # M e r c h a n d i s e r -> Merchandiser

    pattern = r'(?:(?:[A-Za-z]\s){3,}[A-Za-z])'

    def merge(match):
        return match.group(0).replace(" ", "")

    text = re.sub(pattern, merge, text)

    return text


# ==========================================================
# Flags
# ==========================================================

args = set(sys.argv[1:])
do_crawl = "--no-crawl" not in args
do_local = "--only-crawl" not in args


# ==========================================================
# Create Required Directories
# ==========================================================

PDF_DIR.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Load Documents
# ==========================================================

documents = []

if do_local:
    print("\nReading PDF files...")
    for pdf_file in PDF_DIR.glob("*.pdf"):
        print(f"  {pdf_file.name}")
        loader = PyMuPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    print("\nReading Text files...")
    for txt_file in TEXT_DIR.glob("*.txt"):
        print(f"  {txt_file.name}")
        loader = TextLoader(str(txt_file), encoding="utf-8")
        documents.extend(loader.load())


# ==========================================================
# Crawl Websites
# ==========================================================

if do_crawl:
    print("\nCrawling websites...")
    web_docs = crawl_websites()
    print(f"\nCrawled {len(web_docs)} web page(s).")
    documents.extend(web_docs)


if not documents:
    print("\nNo documents found!")
    print("Place PDF/TXT files in:")
    print(f"  {PDF_DIR}")
    print(f"  {TEXT_DIR}")
    print("Or add website URLs to WEBSITE_URLS in config.py.")
    exit()


print(f"\nLoaded {len(documents)} document(s) total.")


print("\nCleaning extracted text...")

for doc in documents:
    doc.page_content = clean_text(doc.page_content)


# ==========================================================
# Split Documents into Chunks
# ==========================================================

print("\nSplitting documents...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")


# ==========================================================
# Create Embeddings
# ==========================================================

print("\nLoading embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)


if CHROMA_DB_DIR.exists():
    print("Removing existing ChromaDB...")
    shutil.rmtree(CHROMA_DB_DIR)


# ==========================================================
# Build ChromaDB
# ==========================================================

print("\nBuilding Chroma_DB...")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=str(CHROMA_DB_DIR)
)




print("ChromaDB created successfully.")
print(f"ChromaDB    : {CHROMA_DB_DIR}")
print(f"Chunks      : {len(chunks)}")
