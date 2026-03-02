"""
Step 1: INGEST DOCUMENTS INTO VECTOR STORE
==========================================
Run this script ONCE to load your documents into ChromaDB.
Run again whenever you add new documents to the docs/ folder.

Usage: python src/ingest.py
"""

import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DOCS_DIR    = "docs"
CHROMA_DIR  = "chroma_db"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE  = 500
CHUNK_OVERLAP = 50


def ingest_documents():
    print("📂 Loading documents from ./docs folder...")
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s)")

    print(f"\n Splitting into chunks (size={CHUNK_SIZE})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"   ✅ Created {len(chunks)} chunks")

    print(f"\n Creating embeddings with Ollama ({EMBED_MODEL})...")
    print("   Make sure Ollama is running: ollama serve")
    print("   Make sure model is pulled: ollama pull nomic-embed-text")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    print(f"\n Saving to ChromaDB at ./{CHROMA_DIR}...")
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print("   (Cleared existing database)")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    print(f"\n Done! {len(chunks)} chunks stored in ChromaDB.")
    print("   You can now run: streamlit run src/app.py")


if __name__ == "__main__":
    ingest_documents()
