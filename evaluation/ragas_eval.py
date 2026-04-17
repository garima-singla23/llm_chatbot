from __future__ import annotations

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

from utils.formatter import build_messages


class RagasEvaluator:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        self.metrics = [faithfulness, answer_relevancy]

    def _retrieve(self, query: str) -> tuple[str, list[dict]]:
        retrieve_fn = getattr(self.retriever, "retrieve", None)
        if callable(retrieve_fn):
            context, sources = retrieve_fn(query)
            return str(context or ""), list(sources or [])

        if callable(self.retriever):
            result = self.retriever(query)
            if isinstance(result, tuple) and len(result) == 2:
                context, sources = result
                return str(context or ""), list(sources or [])
            return str(result or ""), []

        raise TypeError("retriever must be callable or have retrieve(query)")

    def _answer(self, query: str, context: str) -> str:
        system_prompt = (
            "You are an airline customer support specialist. "
            "Answer strictly using the provided policy context."
        )

        # Prefer the signature requested by user if supported.
        try:
            response = self.llm.chat(system_prompt, context, query)
            return str(response)
        except TypeError:
            messages = build_messages(system_prompt, context + "\n\nUser question: " + query)
            response = self.llm.chat(messages, stream=False)
            return str(response)

    def build_dataset(
        self,
        queries: list[str],
        ground_truths: list[str] = None,
    ) -> Dataset:
        answers: list[str] = []
        contexts: list[list[str]] = []

        for query in queries:
            context, _sources = self._retrieve(query)
            answer = self._answer(query, context)
            answers.append(answer)
            contexts.append([context])

        if ground_truths is None:
            ground_truths = [""] * len(queries)

        data = {
            "question": queries,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
        return Dataset.from_dict(data)

    def evaluate(
        self,
        queries: list[str],
        ground_truths: list[str] = None,
    ) -> dict:
        dataset = self.build_dataset(queries, ground_truths)

        metrics = list(self.metrics)
        if ground_truths is not None:
            metrics.append(context_recall)

        result = evaluate(dataset, metrics=metrics)

        output = {
            "faithfulness": float(result["faithfulness"]),
            "answer_relevancy": float(result["answer_relevancy"]),
            "num_queries": len(queries),
        }

        if ground_truths is not None and "context_recall" in result:
            output["context_recall"] = float(result["context_recall"])

        return output
