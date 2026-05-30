("""Build the knowledge base from multiple sources.

Steps:
 1. Wikipedia
 2. Playwright scraping
 3. Kaggle datasets
 4. Hugging Face datasets
 5. Synthetic fallback
 6. Chunk + embed
 7. Summary table
 8. Auto-expand eval set
""")

from pathlib import Path
import json
from datetime import datetime
import pickle


def _banner(step: str):
	print("\n" + "=" * 10 + f" {step} " + "=" * 10)


def _count_files(folder: Path, pattern: str = "*.txt") -> int:
	if not folder.exists():
		return 0
	return len(list(folder.rglob(pattern)))


def build_all():
	base = Path("data/raw")
	base.mkdir(parents=True, exist_ok=True)

	stats = {
		"wikipedia": {"files": 0, "chunks": 0},
		"kaggle": {"files": 0, "chunks": 0},
		"huggingface": {"files": 0, "chunks": 0},
		"playwright": {"files": 0, "chunks": 0},
		"synthetic": {"files": 0, "chunks": 0},
	}

	# Step 1 - Wikipedia
	_banner("Step 1 - Wikipedia")
	try:
		from rag.wiki_scraper import scrape_wiki

		scrape_wiki()
		stats["wikipedia"]["files"] = _count_files(base)
	except Exception as e:
		print("Wikipedia scraping failed:", e)

	# Step 2 - Playwright
	_banner("Step 2 - Playwright scraping")
	try:
		from rag.playwright_scraper import scrape_playwright

		scrape_playwright()
		stats["playwright"]["files"] = _count_files(base)
	except Exception as e:
		print("Playwright scraping failed:", e)

	# Step 3 - Kaggle
	_banner("Step 3 - Kaggle datasets")
	try:
		from rag.kaggle_loader import load_all_kaggle

		try:
			created = load_all_kaggle()
			stats["kaggle"]["files"] = len([p for p in created if p.endswith(".txt")])
		except Exception as inner_e:
			print("Kaggle processing skipped or failed:", inner_e)
	except Exception as e:
		print("Kaggle loader not available, skipping:", e)

	# Step 4 - Hugging Face
	_banner("Step 4 - Hugging Face datasets")
	try:
		from rag.hf_loader import load_all_hf

		try:
			created = load_all_hf()
			stats["huggingface"]["files"] = len([p for p in created if p.endswith(".txt")])
		except Exception as inner_e:
			print("Hugging Face processing skipped or failed:", inner_e)
	except Exception as e:
		print("Hugging Face loader not available, skipping:", e)

	# Step 5 - Synthetic fallback
	_banner("Step 5 - Synthetic fallback")
	try:
		from rag.synthetic_data import generate_synthetic

		# Determine airlines with zero files
		airlines = [
			p.stem.split("_")[-1] for p in base.glob("*.txt")
		]
		# generate_synthetic accepts list of airlines or will generate for defaults
		generated = generate_synthetic(airlines=airlines)
		stats["synthetic"]["files"] = len([p for p in generated if p.endswith(".txt")])
	except Exception as e:
		print("Synthetic generation failed or not available:", e)

	# Step 6 - Chunk + embed
	_banner("Step 6 - Chunking and embedding")
	try:
		from rag.chunker import chunk_all
		from rag.embedder import build_vector_store

		documents = chunk_all()
		chunk_cache_path = Path("data/processed/chunked_documents.pkl")
		chunk_cache_path.parent.mkdir(parents=True, exist_ok=True)
		with chunk_cache_path.open("wb") as f:
			pickle.dump(documents, f)
		build_vector_store(documents)
		stats["wikipedia"]["chunks"] = _count_files(Path("data/vector_store"), "*.bin") if Path("data/vector_store").exists() else 0
	except Exception as e:
		print("Chunking/embedding failed:", e)

	# Step 7 - Summary table
	_banner("Step 7 - Summary")
	# Recompute file counts per source folder heuristically
	stats["wikipedia"]["files"] = _count_files(Path("data/raw"))
	stats["kaggle"]["files"] = _count_files(Path("data/raw"))
	stats["huggingface"]["files"] = _count_files(Path("data/raw"))
	stats["playwright"]["files"] = _count_files(Path("data/raw"))
	stats["synthetic"]["files"] = _count_files(Path("data/raw"))

	total_files = sum(v["files"] for v in stats.values())
	total_chunks = sum(v["chunks"] for v in stats.values())

	print("Source type     | Files | Chunks")
	for k in ["wikipedia", "kaggle", "huggingface", "playwright", "synthetic"]:
		print(f"{k:15} | {stats[k]['files']:5} | {stats[k]['chunks']:6}")
	print(f"{'TOTAL':15} | {total_files:5} | {total_chunks:6}")

	# Step 8 - Auto-expand eval set
	_banner("Step 8 - Auto-expand eval set")
	try:
		from rag.build_eval_from_hf import build_golden_eval_set

		build_golden_eval_set()
	except Exception as e:
		print("Eval expansion failed or build_eval_from_hf not available:", e)


if __name__ == "__main__":
	build_all()

