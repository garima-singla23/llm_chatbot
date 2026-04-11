# test_llm.py
# from llm.factory import get_llm

# llm = get_llm("Mistral(Ollama)")
# print(llm.chat([
#     {"role": "user", "content": "Hello"}
# ]))


from rag.retriever import retrieve_docs
from rag_evaluation.metrics import compute_metrics
from rag_evaluation.evaluation import get_evaluation_data

# Compute MRR
metrics = compute_metrics(get_evaluation_data(), retrieve_docs, k=1)

print("\nFinal MRR Score:", metrics["MRR"])
