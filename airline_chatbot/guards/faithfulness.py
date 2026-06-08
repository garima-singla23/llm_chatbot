from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List


def _tokenize(text: str) -> set[str]:
  return set(re.findall(r"[A-Za-z0-9]+", text.lower()))


def check_faithfulness(answer, docs, threshold=0.3) -> dict:
  answer_tokens = _tokenize(answer if isinstance(answer, str) else str(answer))
  context_tokens = set()
  for doc in docs or []:
    if isinstance(doc, str):
      context_tokens |= _tokenize(doc)
    elif isinstance(doc, dict):
      context_tokens |= _tokenize(str(doc.get("page_content", "")))
    else:
      context_tokens |= _tokenize(str(getattr(doc, "page_content", doc)))

  if not answer_tokens:
    return {"faithful": False, "score": 0.0, "flagged_terms": []}

  overlap = answer_tokens & context_tokens
  score = round(len(overlap) / max(len(answer_tokens), 1), 3)
  flagged_terms = sorted(list(answer_tokens - context_tokens))[:20]
  return {
    "faithful": score >= threshold,
    "score": score,
    "flagged_terms": flagged_terms,
  }
