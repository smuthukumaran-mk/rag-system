"""
crawler.py
----------
Crawls websites starting from seed URLs (same-domain BFS) and returns
LangChain Document objects ready to be embedded.

Used by ingest.py.
"""

import time
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
# from langchain.schema import Document

from config import (
    WEBSITE_URLS,
    MAX_CRAWL_DEPTH,
    MAX_PAGES,
    REQUEST_TIMEOUT,
    USER_AGENT,
)


HEADERS = {"User-Agent": USER_AGENT}
SKIP_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
    ".zip", ".rar", ".mp4", ".mp3", ".doc", ".docx", ".xls", ".xlsx",
    ".ppt", ".pptx", ".ico", ".css", ".js",
)


def _normalize(url: str) -> str:
    url, _ = urldefrag(url)          # drop #fragment
    return url.rstrip("/")


def _same_domain(url: str, root_netloc: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return False
    return netloc == root_netloc or netloc.endswith("." + root_netloc)


def _is_crawlable(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    if parsed.path.lower().endswith(SKIP_EXTENSIONS):
        return False
    return True


def _load_robots(root: str):
    rp = RobotFileParser()
    try:
        parsed = urlparse(root)
        rp.set_url(f"{parsed.scheme}://{parsed.netloc}/robots.txt")
        rp.read()
    except Exception:
        return None
    return rp


def _extract_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")

    # Strip non-content tags
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        tag.decompose()

    title = (soup.title.string.strip()
             if soup.title and soup.title.string else "")

    text = soup.get_text(separator="\n")
    # Collapse whitespace
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return title, "\n".join(lines)


def _extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        absolute = urljoin(base_url, a["href"])
        links.append(_normalize(absolute))
    return links


def crawl_site(seed_url: str) -> list[Document]:
    """BFS crawl a single seed URL, restricted to its own domain."""
    root_netloc = urlparse(seed_url).netloc.lower()
    robots = _load_robots(seed_url)

    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(_normalize(seed_url), 0)])
    documents: list[Document] = []

    while queue and len(documents) < MAX_PAGES:
        url, depth = queue.popleft()

        if url in visited:
            continue
        visited.add(url)

        if not _is_crawlable(url) or not _same_domain(url, root_netloc):
            continue

        if robots and not robots.can_fetch(USER_AGENT, url):
            print(f"  [robots] skip {url}")
            continue

        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
        except Exception as e:
            print(f"  [error] {url} -> {e}")
            continue

        ctype = resp.headers.get("Content-Type", "")
        if "text/html" not in ctype.lower():
            continue

        title, text = _extract_text(resp.text)

        if len(text) < 100:
            # skip near-empty pages
            pass
        else:
            print(f"  [{len(documents)+1}/{MAX_PAGES}] depth={depth}  {url}")
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": url,
                        "title": title,
                        "type": "web",
                    },
                )
            )

        if depth < MAX_CRAWL_DEPTH:
            for link in _extract_links(resp.text, url):
                if link not in visited:
                    queue.append((link, depth + 1))

        time.sleep(0.5)  # be polite

    return documents


def crawl_websites(seed_urls: list[str] = None) -> list[Document]:
    """Crawl every seed URL and return the combined document list."""
    seeds = seed_urls if seed_urls is not None else WEBSITE_URLS
    all_docs: list[Document] = []

    for seed in seeds:
        print(f"\nCrawling: {seed}")
        docs = crawl_site(seed)
        print(f"  -> {len(docs)} page(s) from {seed}")
        all_docs.extend(docs)

    return all_docs


if __name__ == "__main__":
    docs = crawl_websites()
    print(f"\nTotal crawled documents: {len(docs)}")
