"""
config.py
---------
Central configuration file for the College Information Assistant.
"""

from pathlib import Path

# ==========================================================
# Project Directories
# ==========================================================

# Root project folder
BASE_DIR = Path(__file__).resolve().parent

# Data folders
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdf"
TEXT_DIR = DATA_DIR / "text"

WEBSITE_URLS = []

# ==========================================================
# Website Crawling
# ==========================================================

MAX_CRAWL_DEPTH = 5
MAX_PAGES = 400
REQUEST_TIMEOUT = 15

USER_AGENT = (
    "College-RAG-Bot/1.0 (Educational Project)"
)


# Embedding storage
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# ==========================================================
# Ollama Configuration
# ==========================================================

OLLAMA_MODEL = "qwen2.5:7b"

# ==========================================================
# Embedding Model
# ==========================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ==========================================================
# Text Splitting
# ==========================================================

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3