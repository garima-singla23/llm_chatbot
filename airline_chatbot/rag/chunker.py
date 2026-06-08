from pathlib import Path
import json

try:
	from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
	# Minimal fallback splitter: fixed-size chunks with overlap
	class RecursiveCharacterTextSplitter:
		def __init__(self, chunk_size=400, chunk_overlap=60):
			self.chunk_size = chunk_size
			self.chunk_overlap = chunk_overlap

		def split_text(self, text: str):
			if not text:
				return []
			chunks = []
			start = 0
			while start < len(text):
				end = start + self.chunk_size
				chunks.append(text[start:end])
				start = end - self.chunk_overlap
				if start < 0:
					start = 0
			return chunks

try:
	from langchain.schema import Document
except Exception:
	# Lightweight Document substitute
	class Document:
		def __init__(self, page_content, metadata=None):
			self.page_content = page_content
			self.metadata = metadata or {}


TOPIC_KEYWORDS = {
	"baggage": [
		"baggage",
		"luggage",
		"cabin bag",
		"check-in bag",
		"carry-on",
		"excess baggage",
	],
	"refund": [
		"refund",
		"cancellation",
		"cancel",
		"reschedule",
		"rescheduling",
		"credit shell",
	],
	"check_in": [
		"check-in",
		"check in",
		"web check-in",
		"boarding pass",
		"boarding",
	],
	"visa": [
		"visa",
		"passport",
		"immigration",
		"travel documents",
		"entry requirement",
	],
	"seat": [
		"seat",
		"seat selection",
		"legroom",
		"aisle",
		"window",
	],
	"meal": [
		"meal",
		"food",
		"snack",
		"beverage",
		"special meal",
	],
	"delay": [
		"delay",
		"delayed",
		"disruption",
		"rescheduled",
		"missed connection",
	],
	"general": [
		"policy",
		"information",
		"support",
		"help",
		"faq",
	],
}


def detect_topic(text):
	lowered = text.lower()
	for topic, keywords in TOPIC_KEYWORDS.items():
		if topic == "general":
			continue
		for keyword in keywords:
			if keyword in lowered:
				return topic
	return "general"


def detect_route_type(text):
	lowered = text.lower()
	domestic_keywords = ["domestic", "within india", "intra-india", "regional"]
	international_keywords = [
		"international",
		"overseas",
		"foreign",
		"outside india",
	]

	has_domestic = any(keyword in lowered for keyword in domestic_keywords)
	has_international = any(keyword in lowered for keyword in international_keywords)

	if has_domestic and not has_international:
		return "domestic"
	if has_international and not has_domestic:
		return "international"
	return "both"


def chunk_file(filepath, metadata_override=None):
	file_path = Path(filepath)
	if not file_path.exists():
		print(f"[WARN] File not found: {file_path}")
		return []

	text = file_path.read_text(encoding="utf-8")

	meta_path = file_path.with_suffix(".meta.json")
	metadata = {}
	if meta_path.exists():
		try:
			metadata = json.loads(meta_path.read_text(encoding="utf-8"))
		except json.JSONDecodeError:
			print(f"[WARN] Invalid JSON metadata: {meta_path}")

	if metadata_override:
		metadata.update(metadata_override)

	splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
	chunks = splitter.split_text(text)

	documents = []
	for idx, chunk in enumerate(chunks, start=1):
		doc_metadata = {
			"airline": metadata.get("airline", "unknown"),
			"source_url": metadata.get("source_url", ""),
			"chunk_id": f"{file_path.stem}_{idx}",
			"topic": detect_topic(chunk),
			"route_type": detect_route_type(chunk),
			"char_count": len(chunk),
		}
		documents.append(Document(page_content=chunk, metadata=doc_metadata))

	print(f"[INFO] {file_path.name}: {len(documents)} chunks")
	return documents


def filter_chunks(documents, max_total=5000):
	import hashlib, random
	from collections import defaultdict

	print(f"[FILTER] Input: {len(documents)} chunks")

	# Deduplicate by first 120 chars fingerprint
	seen, deduped = set(), []
	for doc in documents:
		fp = hashlib.md5(doc.page_content[:120].lower().strip().encode()).hexdigest()
		if fp not in seen:
			seen.add(fp)
			deduped.append(doc)
	print(f"[FILTER] After dedup: {len(deduped)}")

	# Quality filter
	keywords = ['flight','airline','baggage','luggage','refund','cancel',
	            'check','book','seat','ticket','airport','travel','passenger',
	            'depart','arriv','policy','fee','kg','allow','indigo',
	            'spicejet','vistara','air india']
	quality = [
	    d for d in deduped
	    if len(d.page_content.strip()) >= 80
	    and sum(c.isalpha() for c in d.page_content) / max(len(d.page_content),1) >= 0.4
	    and any(kw in d.page_content.lower() for kw in keywords)
	]
	print(f"[FILTER] After quality: {len(quality)}")

	# Proportional sample if still over limit
	if len(quality) > max_total:
		by_source = defaultdict(list)
		for doc in quality:
			by_source[doc.metadata.get("source_type","unknown")].append(doc)
		sampled = []
		for src, docs in by_source.items():
			alloc = max(100, int(max_total * len(docs) / len(quality)))
			random.seed(42)
			selected = random.sample(docs, min(alloc, len(docs)))
			for doc in selected:
				assert doc.metadata, f"Metadata missing on chunk: {doc.page_content[:40]}"
			sampled.extend(selected)
		quality = sampled[:max_total]
	print(f"[FILTER] Final: {len(quality)} chunks")
	# Debug check: print metadata for a small sample to ensure metadata preserved
	sample = quality[:3]
	for doc in sample:
		print(f"[META CHECK] airline={doc.metadata.get('airline','MISSING')} "
			f"source={doc.metadata.get('source_type','MISSING')} "
			f"topic={doc.metadata.get('topic','MISSING')}")

	# Verify metadata survived for a few items before returning
	sample2 = quality[:5]
	for doc in sample2:
		airline = doc.metadata.get('airline', 'MISSING')
		topic = doc.metadata.get('topic', 'MISSING')
		if airline == 'MISSING':
			print(f"[META WARN] Metadata lost: {doc.page_content[:50]}")
	return quality


def chunk_all(raw_dir="data/raw"):
	# Check first raw document metadata
	test_docs = []
	for filepath in list(Path(raw_dir).glob("*.txt"))[:1]:
		test_docs = chunk_file(str(filepath))
		if test_docs:
			print(f"[META DEBUG] First chunk metadata: {test_docs[0].metadata}")
			break

	raw_path = Path(raw_dir)
	txt_files = sorted(raw_path.glob("*.txt"))

	all_documents = []
	for txt_file in txt_files:
		file_docs = chunk_file(txt_file)
		all_documents.extend(file_docs)

	print(f"[INFO] Total chunks: {len(all_documents)}")
	all_documents = filter_chunks(all_documents, max_total=15000)
	from collections import Counter
	src_counts = Counter(d.metadata.get("source_type","unknown") for d in all_documents)
	for src, count in sorted(src_counts.items()):
		print(f"[FILTER] {src}: {count} chunks")
	return all_documents
