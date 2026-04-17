import os

from rag.retriever import build_vectorstore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.join(BASE_DIR, "rag")

os.makedirs(RAG_DIR, exist_ok=True)

build_vectorstore()
print("✅ Vector store built successfully")

chunk_metadata_path = os.path.join(RAG_DIR, "chunk_metadata.json")
if os.path.exists(chunk_metadata_path):
    print(f"✅ Chunk metadata saved: {chunk_metadata_path}")
