from __future__ import annotations

import json
import re

from utils.formatter import build_messages

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for an airline policy and flight search
chatbot. You score answers on three dimensions, each 1-5:

ACCURACY: Is the answer factually correct based on the context provided?
1=completely wrong, 3=partially correct, 5=fully correct

GROUNDEDNESS: Is every claim in the answer supported by the context?
1=mostly hallucinated, 3=mixed, 5=fully grounded in context

HELPFULNESS: Does the answer actually help the user with their question?
1=not helpful, 3=somewhat helpful, 5=very helpful

Respond ONLY with valid JSON, no explanation:
{"accuracy": N, "groundedness": N, "helpfulness": N, "reasoning": "..."}
"""


class LLMJudge:
    def __init__(self, llm):
        self.llm = llm

    @staticmethod
    def _parse_score_payload(raw_text: str) -> dict:
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", raw_text)
            if not match:
                raise
            payload = json.loads(match.group(0))

        return {
            "accuracy": int(payload.get("accuracy", 0)),
            "groundedness": int(payload.get("groundedness", 0)),
            "helpfulness": int(payload.get("helpfulness", 0)),
            "reasoning": str(payload.get("reasoning", "")),
        }

    def score_answer(self, query: str, context: str, answer: str) -> dict:
        user_message = (
            f"Query: {query}\n\n"
            f"Context used:\n{context[:1500]}\n\n"
            f"Answer:\n{answer}"
        )

        try:
            raw_response = self.llm.chat(JUDGE_SYSTEM_PROMPT, "", user_message)
        except TypeError:
            messages = build_messages(JUDGE_SYSTEM_PROMPT, user_message)
            raw_response = self.llm.chat(messages, stream=False)

        try:
            return self._parse_score_payload(str(raw_response))
        except Exception:
            return {
                "accuracy": 0,
                "groundedness": 0,
                "helpfulness": 0,
                "reasoning": "parse_error",
            }

    def score_batch(
        self,
        queries: list[str],
        contexts: list[str],
        answers: list[str],
    ) -> list[dict]:
        scores: list[dict] = []

        for i, (query, context, answer) in enumerate(zip(queries, contexts, answers)):
            print(f"[Judge] Scoring {i+1}/{len(queries)}...")
            scores.append(self.score_answer(query, context, answer))

        return scores

    def summary(self, scores: list[dict]) -> dict:
        valid_scores = [s for s in scores if s.get("reasoning") != "parse_error"]
        n_errors = len(scores) - len(valid_scores)

        if not valid_scores:
            return {
                "mean_accuracy": 0.0,
                "mean_groundedness": 0.0,
                "mean_helpfulness": 0.0,
                "overall": 0.0,
                "n_scored": 0,
                "n_errors": n_errors,
            }

        mean_accuracy = sum(float(s.get("accuracy", 0)) for s in valid_scores) / len(valid_scores)
        mean_groundedness = sum(float(s.get("groundedness", 0)) for s in valid_scores) / len(valid_scores)
        mean_helpfulness = sum(float(s.get("helpfulness", 0)) for s in valid_scores) / len(valid_scores)
        overall = (mean_accuracy + mean_groundedness + mean_helpfulness) / 3

        return {
            "mean_accuracy": mean_accuracy,
            "mean_groundedness": mean_groundedness,
            "mean_helpfulness": mean_helpfulness,
            "overall": overall,
            "n_scored": len(valid_scores),
            "n_errors": n_errors,
        }
