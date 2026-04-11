import os

from rag.retriever import build_vectorstore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.join(BASE_DIR, "rag")

os.makedirs(RAG_DIR, exist_ok=True)

build_vectorstore()
print("✅ Vector store built successfully")
