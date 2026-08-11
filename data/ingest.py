# ==============================================================
# data/ingest.py — Document Ingestion Service
# ==============================================================
import os
import sys
import hashlib
from pathlib import Path
from typing import List

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from rich.console import Console

from infrastructure.retrieval.embeddings import get_embeddings
from core.config import settings

console = Console()

# ── Settings ───────────────────────────────────────────────────
DOCUMENTS_DIR = Path(__file__).parent / "documents"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150


def _load_single_file(file_path: Path) -> List[Document]:
    """Helper to load a single file based on its extension."""
    if file_path.suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    else:
        loader = TextLoader(str(file_path), encoding="utf-8")
    return loader.load()


def process_documents(docs: List[Document]) -> int:
    """
    CORE LOGIC: Takes raw documents, splits them, hashes them, 
    and saves them to ChromaDB. 
    (Used by both batch ingestion and single-file API).
    """
    if not docs:
        return 0

    # 1. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    # Clean text to avoid encoding errors from PDFs
    for chunk in chunks:
        chunk.page_content = chunk.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")

    # 2. Hash and Deduplicate BEFORE embedding
    vector_store = Chroma(
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_db_path
    )
    
    # Get all existing IDs currently in the database
    existing_data = vector_store.get(include=[])
    existing_ids = set(existing_data["ids"]) if existing_data and "ids" in existing_data else set()

    # Filter out chunks that we've already embedded before
    new_chunks = []
    new_ids = []
    
    for chunk in chunks:
        chunk_hash = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()
        if chunk_hash not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(chunk_hash)

    if not new_chunks:
        console.print("[dim]No new content to embed. Everything is already in the database![/dim]")
        return 0

    # 3. Embed and Save ONLY the strictly new chunks
    console.print(f"Sending {len(new_chunks)} purely new chunks to be embedded...")
    vector_store.add_documents(documents=new_chunks, ids=new_ids)
    
    return len(new_chunks)


def ingest_file(file_path: str | Path) -> int:
    """
    API ENTRY POINT: Call this from a backend server when a new file is uploaded.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find file: {path}")
    
    docs = _load_single_file(path)
    return process_documents(docs)


def ingest_all():
    """
    CLI ENTRY POINT: Scans the entire data/documents folder and processes everything.
    """
    console.rule("[bold blue]Imi AI — Document Ingestion[/bold blue]")
    
    files = list(DOCUMENTS_DIR.glob("**/*.pdf")) + list(DOCUMENTS_DIR.glob("**/*.txt"))
    if not files:
        console.print(f"[yellow]No documents found in {DOCUMENTS_DIR}[/yellow]")
        return

    all_docs = []
    for file in files:
        try:
            all_docs.extend(_load_single_file(file))
            console.print(f"   Loaded: [cyan]{file.name}[/cyan]")
        except Exception as e:
            console.print(f"   Failed: [red]{file.name}[/red] — {e}")

    chunks_created = process_documents(all_docs)
    
    console.rule("[bold green] Ingestion Complete![/bold green]")
    console.print(f"  Documents indexed: [green]{chunks_created} chunks[/green]")
    console.print(f"  Stored at: [cyan]{settings.chroma_db_path}[/cyan]")


if __name__ == "__main__":
    ingest_all()
