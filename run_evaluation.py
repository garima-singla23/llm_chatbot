from rag_evaluation.metrics import compute_metrics
from rag_evaluation.evaluation import get_evaluation_data
from rag_evaluation.experiment_logger import log_experiment
from rag.retriever import retrieve_docs

evaluation_data = get_evaluation_data()
mrr = compute_metrics(evaluation_data, retrieve_docs, k=3)

results = {
    "chunk_size": 500,
    "overlap": 50,
    "top_k": 3,
    "MRR": mrr
}

log_experiment(results)

print("Evaluation Complete:", results)
