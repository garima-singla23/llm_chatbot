from __future__ import annotations

from utils.formatter import build_messages

HYDE_SYSTEM_PROMPT = """You are an airline policy expert. Given a user
question, write a SHORT hypothetical answer (2-3 sentences) as if you
were reading directly from an airline policy document. Do not say you
don't know. Write factual-sounding policy text even if approximate.
This text will be used for document retrieval, not shown to the user."""


def generate_hypothetical_answer(query: str, llm) -> str:
    try:
        try:
            response = llm.chat(HYDE_SYSTEM_PROMPT, "", query)
        except TypeError:
            messages = build_messages(HYDE_SYSTEM_PROMPT, query)
            response = llm.chat(messages, stream=False)

        print(f"[HyDE] Generated hypothesis for: {query[:60]}")
        return str(response)
    except Exception:
        return query


def _retrieve(retriever, query: str) -> tuple[str, list[dict]]:
    retrieve_fn = getattr(retriever, "retrieve", None)
    if callable(retrieve_fn):
        context, sources = retrieve_fn(query)
        return str(context or ""), list(sources or [])

    if callable(retriever):
        result = retriever(query)
        if isinstance(result, tuple) and len(result) == 2:
            context, sources = result
            return str(context or ""), list(sources or [])
        return str(result or ""), []

    raise TypeError("retriever must be callable or expose retrieve(query)")


def hyde_retrieve(
    query: str,
    llm,
    retriever,
    use_hyde: bool = True,
) -> tuple[str, list[dict]]:
    if not use_hyde:
        return _retrieve(retriever, query)

    hypothesis = generate_hypothetical_answer(query, llm)
    context, sources = _retrieve(retriever, hypothesis)
    return context, sources
