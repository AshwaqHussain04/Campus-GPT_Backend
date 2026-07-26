"""
Ingest documents from app/data/ into MongoDB Atlas Vector Search + local docstore.

Only processes files that are new or have changed since the last run
(tracked via app/storage/manifest.json, keyed by content hash).

PDFs and .txt files are loaded manually via pypdf rather than through
LlamaIndex's SimpleDirectoryReader, which was found to inflate PDF text
extraction by ~180x on real documents (likely pulling in embedded
font/image data as text alongside the actual page content).

Usage:
    python -m app.ingest
"""
import hashlib
import json
import os
import sys
import time

import pypdf
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.storage.docstore import SimpleDocumentStore

from app.db import ping
from app.rag_setup import configure_llama_settings, get_vector_store

DATA_DIR = "app/data"
STORAGE_DIR = "app/storage"
MANIFEST_PATH = os.path.join(STORAGE_DIR, "manifest.json")


def file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict) -> None:
    os.makedirs(STORAGE_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def find_new_or_changed_files(manifest: dict) -> tuple[list[str], dict]:
    """Compare files in DATA_DIR against the manifest. Returns (paths_to_ingest, updated_manifest)."""
    updated = dict(manifest)
    to_ingest = []

    for root, _, files in os.walk(DATA_DIR):
        for fname in files:
            path = os.path.join(root, fname)
            current_hash = file_hash(path)
            if manifest.get(path) != current_hash:
                to_ingest.append(path)
                updated[path] = current_hash

    return to_ingest, updated


def load_documents_manually(paths: list[str]) -> list[Document]:
    """Load .txt and .pdf files ourselves instead of using SimpleDirectoryReader."""
    documents = []
    for path in paths:
        file_name = os.path.basename(path)

        if path.lower().endswith(".pdf"):
            reader = pypdf.PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif path.lower().endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            print(f"Skipping unsupported file type: {path}")
            continue

        documents.append(Document(text=text, metadata={"file_name": file_name}))
        print(f"Loaded {file_name}: {len(text)} characters")

    return documents


def load_docstore() -> SimpleDocumentStore:
    docstore_path = os.path.join(STORAGE_DIR, "docstore.json")
    if os.path.exists(docstore_path):
        return SimpleDocumentStore.from_persist_dir(STORAGE_DIR)
    return SimpleDocumentStore()


def ingest_with_retry(documents, storage_context, max_retries=5):
    for attempt in range(max_retries):
        try:
            return VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True,
                store_nodes_override=True,
            )
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                wait = 60 * (attempt + 1)
                print(f"Rate limited. Waiting {wait}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded due to rate limiting.")


def main():
    if not ping():
        print("Aborting: could not connect to MongoDB. Check MONGODB_URI in .env.local")
        sys.exit(1)

    configure_llama_settings()

    manifest = load_manifest()
    paths_to_ingest, updated_manifest = find_new_or_changed_files(manifest)

    if not paths_to_ingest:
        print("No new or changed files. Nothing to ingest.")
        return

    print(f"Found {len(paths_to_ingest)} new/changed file(s): {paths_to_ingest}")

    documents = load_documents_manually(paths_to_ingest)

    docstore = load_docstore()
    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(
        docstore=docstore,
        vector_store=vector_store,
    )

    print("Embedding chunks and writing to MongoDB Atlas + local docstore...")
    ingest_with_retry(documents, storage_context)

    os.makedirs(STORAGE_DIR, exist_ok=True)
    docstore.persist(persist_path=os.path.join(STORAGE_DIR, "docstore.json"))
    save_manifest(updated_manifest)

    print("Ingestion complete.")


if __name__ == "__main__":
    main()