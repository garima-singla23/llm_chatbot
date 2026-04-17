from llm.factory import get_llm
from rag.retriever import retrieve_docs
from rag_evaluation.metrics import compute_metrics
from rag_evaluation.evaluation import get_evaluation_data
from rag_evaluation.experiment_logger import log_experiment
from evaluation.ragas_eval import RagasEvaluator
from evaluation.ragas_queries import RAGAS_QUERIES
from evaluation.judge import LLMJudge
from evaluation.flight_eval import FlightEvalHarness
from utils.formatter import build_messages


class _EvalDoc:
    def __init__(self, chunk_id):
        self.metadata = {"chunk_id": chunk_id}


class _RetrieverAdapter:
    def retrieve(self, query: str):
        return retrieve_docs(query)


def _retriever_for_metrics(query, k=3):
    _context, sources = retrieve_docs(query, k=k, rerank=True, expand_window=False)
    filtered = [s for s in sources if not bool(s.get("expanded", False))]
    return [_EvalDoc(source.get("chunk_id")) for source in filtered[:k]]


evaluation_data = get_evaluation_data()
retrieval_metrics = compute_metrics(evaluation_data, _retriever_for_metrics, k=3)

llm = get_llm("Phi-3 (Ollama)")
retriever = _RetrieverAdapter()

ragas_eval = RagasEvaluator(llm=llm, retriever=retriever)
ragas_results = ragas_eval.evaluate(RAGAS_QUERIES)

judge = LLMJudge(llm=llm)
queries = RAGAS_QUERIES[:10]
contexts, answers = [], []
system_prompt = (
    "You are an airline customer support specialist. "
    "Answer strictly using the provided policy context."
)

for q in queries:
    ctx, _ = retriever.retrieve(q)
    try:
        ans = llm.chat(system_prompt, ctx, q)
    except TypeError:
        messages = build_messages(system_prompt, ctx + "\n\nUser question: " + q)
        ans = llm.chat(messages, stream=False)
    contexts.append(str(ctx))
    answers.append(str(ans))

scores = judge.score_batch(queries, contexts, answers)
summary = judge.summary(scores)

harness = FlightEvalHarness(llm=llm)
flight_results = harness.run()

results = {
    "chunk_size": 500,
    "overlap": 50,
    "top_k": 3,
    "MRR": retrieval_metrics["MRR"],
    "Precision@K": retrieval_metrics["Precision@K"],
    "Recall@K": retrieval_metrics["Recall@K"],
    "faithfulness": ragas_results["faithfulness"],
    "answer_relevancy": ragas_results["answer_relevancy"],
    "judge_accuracy": summary["mean_accuracy"],
    "judge_groundedness": summary["mean_groundedness"],
    "judge_helpfulness": summary["mean_helpfulness"],
    "judge_overall": summary["overall"],
    "entity_full_match_rate": flight_results["full_match_rate"],
    "entity_field_accuracy": flight_results["mean_field_accuracy"],
}

log_experiment(results)

print(f"MRR:               {retrieval_metrics['MRR']:.3f}")
print(f"Precision@K:       {retrieval_metrics['Precision@K']:.3f}")
print(f"Recall@K:          {retrieval_metrics['Recall@K']:.3f}")
print(f"Faithfulness:      {ragas_results['faithfulness']:.3f}")
print(f"Answer Relevancy:  {ragas_results['answer_relevancy']:.3f}")
print(f"Judge overall: {summary['overall']:.2f}/5")
print(f"Accuracy:      {summary['mean_accuracy']:.2f}")
print(f"Groundedness:  {summary['mean_groundedness']:.2f}")
print(f"Helpfulness:   {summary['mean_helpfulness']:.2f}")
print(f"Entity extraction: {flight_results['full_match_rate']:.0%} perfect")
print(f"Field accuracy:    {flight_results['mean_field_accuracy']:.0%}")
