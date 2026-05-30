from rag.embedder import load_vector_store
from rag.chunker import chunk_all
from rag.retriever import build_hybrid_retriever, retrieve

print("Loading vector store...")
vs = load_vector_store()

print("Loading documents...")
docs = chunk_all()

print("Building retriever...")
retriever = build_hybrid_retriever(vs, docs)

print("Testing retrieval...")
results = retrieve(
    "IndiGo baggage limit domestic",
    retriever,
    airline_filter="indigo"
)

print("Retrieved", len(results), "chunks")

for r in results:
    airline = r.metadata.get("airline", "?")
    print(f"[{airline}] {r.page_content[:80]}...")

print("SUCCESS — vector store is working")
