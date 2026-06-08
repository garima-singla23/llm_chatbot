import json
from pathlib import Path

from dotenv import load_dotenv

# Ensure environment variables (e.g., XAI_API_KEY) are loaded for eval context.
# The workspace .env sits one level above the repo root.
workspace_env = Path(__file__).resolve().parents[2] / ".env"
if workspace_env.exists():
    load_dotenv(workspace_env)
else:
    load_dotenv()
import os

api_key = os.getenv("GROQ_API_KEY") or os.getenv("XAI_API_KEY")
assert api_key, "GROQ_API_KEY or XAI_API_KEY not found in environment"

from chatbot import chain as chain_module
from chatbot import memory as memory_module
from rag import embedder as embedder_module
from rag import chunker as chunker_module
from rag import retriever as retriever_module


def load_golden_set(golden_set_path: str) -> list:
    """Load the golden set of evaluation cases from JSON file."""
    with open(golden_set_path, "r") as f:
        return json.load(f)


def check_keywords(text, keywords: list) -> tuple:
    """
    Check which keywords appear in the text (case-insensitive).
    Handles keywords that may be provided as dicts or strings.
    Returns (number of hits, total keywords).
    """
    text_lower = (str(text) if text is not None else "").lower().strip()

    # Normalize keywords to a list of strings
    norm_keys = []
    for k in (keywords or []):
        if isinstance(k, str):
            norm_keys.append(k.strip())
        elif isinstance(k, dict):
            # try common fields
            if "keyword" in k:
                norm_keys.append(str(k.get("keyword", "")).strip())
            elif "text" in k:
                norm_keys.append(str(k.get("text", "")).strip())
            else:
                norm_keys.append(str(k))
        else:
            norm_keys.append(str(k))

    hits = 0
    for keyword in norm_keys:
        if not keyword:
            continue
        if keyword.lower() in text_lower:
            hits += 1
    return hits, len(norm_keys)


def run_evals(retriever, golden_set: list) -> dict:
    """
    Run evaluation on all golden set cases.
    Returns a dict with pass/fail results and overall metrics.
    """
    results = []
    total_keywords = 0
    total_hits = 0

    # basic sanity check
    assert retriever is not None, "Retriever is None"
    if len(golden_set) > 0:
        print("DEBUG first case:", golden_set[0])

    for idx, case in enumerate(golden_set, 1):
        question = case["question"]
        expected_keywords = case["expected_keywords"]
        airline_filter = case.get("airline")

        # Create fresh memory for each case
        memory = memory_module.ConversationMemory()

        # Call chat with the question
        try:
            answer = chain_module.chat(
                question,
                memory,
                retriever,
                airline_filter=airline_filter
            )
        except Exception as e:
            answer = f"Error: {str(e)}"

        # DEBUG: print the raw answer (first case and empties)
        print(f"DEBUG answer: '{(str(answer)[:100] if answer else 'EMPTY')}'")
        if not answer or len(str(answer).strip()) == 0:
            print(f"ERROR: chat() returned empty for: {question}")
            # skip keyword checking for empty answers
            results.append({
                "idx": idx,
                "status": "FAIL",
                "question": question,
                "score": 0,
                "answer": answer,
                "case": case
            })
            total_keywords += len(expected_keywords or [])
            continue

        # Check which keywords appear in the answer
        hits, total = check_keywords(answer, expected_keywords)
        total_keywords += total
        total_hits += hits

        # Calculate keyword score
        keyword_score = (hits / total * 100) if total > 0 else 0

        # Determine PASS/FAIL based on keyword score (threshold: 50%)
        status = "PASS" if keyword_score >= 50 else "FAIL"

        # Truncate question to 55 chars
        question_truncated = (question[:55] + "...") if len(question) > 55 else question

        results.append({
            "idx": idx,
            "status": status,
            "question": question_truncated,
            "score": keyword_score,
            "answer": answer,
            "case": case
        })

    return {
        "results": results,
        "total_keywords": total_keywords,
        "total_hits": total_hits,
        "passed": sum(1 for r in results if r["status"] == "PASS"),
        "total": len(results)
    }


def print_results(eval_results: dict):
    """Print evaluation results in a formatted table."""
    results = eval_results["results"]
    passed = eval_results["passed"]
    total = eval_results["total"]
    total_keywords = eval_results["total_keywords"]
    total_hits = eval_results["total_hits"]

    print("\n" + "=" * 90)
    print("EVALUATION RESULTS — Airline Chatbot Golden Set")
    print("=" * 90 + "\n")

    # Print table header
    print(f"{'#':<3} {'Status':<7} {'Question':<56} {'Score%':<8}")
    print("-" * 90)

    # Print each result
    for result in results:
        idx = result["idx"]
        status = result["status"]
        question = result["question"]
        score = result["score"]

        print(f"{idx:<3} {status:<7} {question:<56} {score:>6.1f}%")

    print("-" * 90)
    print(f"\n📊 Summary:")
    print(f"   Passed: {passed}/{total}")
    print(f"   Pass Rate: {(passed/total*100):.1f}%")
    print(f"   Overall Keyword Hit Rate: {(total_hits/total_keywords*100):.1f}% ({total_hits}/{total_keywords})")
    print("\n" + "=" * 90 + "\n")


if __name__ == "__main__":
    # Paths
    evals_dir = Path(__file__).parent
    golden_set_path = evals_dir / "golden_set.json"
    project_root = evals_dir.parent

    # Load golden set
    print("Loading golden set...")
    golden_set = load_golden_set(str(golden_set_path))
    print(f"✓ Loaded {len(golden_set)} test cases")

    # Load vector store and build retriever
    print("\nBuilding retriever...")
    try:
        vs = embedder_module.load_vector_store()
        docs = chunker_module.chunk_all(str(project_root / "data" / "raw"))
        retriever = retriever_module.build_hybrid_retriever(vs, docs)
        print("✓ Retriever built successfully")
    except Exception as e:
        print(f"✗ Failed to build retriever: {e}")
        exit(1)

    # Run evaluations
    print(f"\nRunning {len(golden_set)} evaluation cases...\n")
    eval_results = run_evals(retriever, golden_set)

    # Print results
    print_results(eval_results)
