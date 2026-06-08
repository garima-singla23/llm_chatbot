import json
import gc
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


BATCH_SIZE = 16  # embed 16 chunks at a time, save after each batch


def get_embeddings(use_openai=False):
	if use_openai:
		return OpenAIEmbeddings(model="text-embedding-3-small")
	return HuggingFaceEmbeddings(
		model_name="sentence-transformers/all-MiniLM-L6-v2",
		model_kwargs={"device": "cpu"},
		encode_kwargs={"batch_size": 16},
	)


def build_vector_store(documents, save_path="data/vector_store"):
	"""Build or resume building a FAISS vector store with checkpoint support."""
	Path(save_path).mkdir(parents=True, exist_ok=True)
	embeddings = get_embeddings()
	checkpoint_path = Path(save_path) / "checkpoint.json"

	# --- Resume logic ---
	start_idx = 0
	vectorstore = None
	if checkpoint_path.exists():
		with open(checkpoint_path) as f:
			ckpt = json.load(f)
		start_idx = ckpt.get("embedded_count", 0)
		print(f"[RESUME] Resuming from chunk {start_idx}/{len(documents)}")
		try:
			vectorstore = FAISS.load_local(
				save_path, embeddings,
				allow_dangerous_deserialization=True
			)
			print(f"[RESUME] Loaded existing FAISS index")
		except Exception as e:
			start_idx = 0
			vectorstore = None
			print(f"[RESUME] Could not load index — starting fresh: {e}")

	total = len(documents)
	remaining = documents[start_idx:]

	# Ensure each Document has an `id` attribute (FAISS/langchain expects it).
	for i, d in enumerate(documents):
		if not hasattr(d, "id"):
			try:
				setattr(d, "id", d.metadata.get("chunk_id") if isinstance(d.metadata, dict) else None or f"doc_{i}")
			except Exception:
				# If the Document is immutable, continue — FAISS may accept it anyway.
				pass

	if not remaining:
		print(f"[INFO] All {total} chunks already embedded.")
		return vectorstore

	print(f"[INFO] Embedding {len(remaining)} remaining chunks in batches of {BATCH_SIZE}...")

	for batch_start in range(0, len(remaining), BATCH_SIZE):
		batch = remaining[batch_start: batch_start + BATCH_SIZE]
		batch_num = (batch_start // BATCH_SIZE) + 1
		total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE

		try:
			# Ensure batch docs have ids as well (defensive)
			for j, d in enumerate(batch):
				if not hasattr(d, "id"):
					try:
						setattr(d, "id", d.metadata.get("chunk_id") if isinstance(d.metadata, dict) else None or f"batch_{batch_start}_{j}")
					except Exception:
						pass
			if vectorstore is None:
				vectorstore = FAISS.from_documents(batch, embeddings)
			else:
				batch_store = FAISS.from_documents(batch, embeddings)
				vectorstore.merge_from(batch_store)
				del batch_store

			# Save index + checkpoint after every batch
			vectorstore.save_local(save_path)
			embedded_so_far = start_idx + batch_start + len(batch)
			with open(checkpoint_path, "w") as f:
				json.dump({"embedded_count": embedded_so_far, "total": total}, f)

			print(f"[INFO] Batch {batch_num}/{total_batches} done — {embedded_so_far}/{total} chunks saved")

			# Free memory
			gc.collect()

		except MemoryError as e:
			print(f"[ERROR] Memory error at batch {batch_num}. Progress saved at chunk {start_idx + batch_start}.")
			print(f"[INFO] Re-run build_knowledge_base.py — it will resume from here.")
			raise

	# Clear checkpoint when fully done
	if checkpoint_path.exists():
		checkpoint_path.unlink()
	print(f"[DONE] All {total} chunks embedded and saved to {save_path}")
	return vectorstore


def load_vector_store(path="data/vector_store"):
	embeddings = get_embeddings()
	return FAISS.load_local(
		path,
		embeddings,
		allow_dangerous_deserialization=True,
	)


def verify_vector_store(path="data/vector_store"):
	"""Load the store and check metadata on 10 random docs."""
	from collections import Counter

	embeddings = get_embeddings()
	vs = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

	# Sample 10 docs and check metadata
	docs = vs.similarity_search("IndiGo baggage", k=10)
	print("\n[VERIFY] Metadata check on 10 retrieved docs:")
	for doc in docs:
		airline = doc.metadata.get('airline', 'MISSING')
		topic = doc.metadata.get('topic', 'MISSING')
		print(f"  airline={airline} topic={topic} | {doc.page_content[:60]}...")

	# Count by airline
	all_docs = vs.similarity_search("flight airline", k=100)
	airlines = Counter(d.metadata.get('airline','MISSING') for d in all_docs)
	print(f"\n[VERIFY] Airline distribution in sample: {dict(airlines)}")
	return vs


if __name__ == "__main__":
	verify_vector_store()
