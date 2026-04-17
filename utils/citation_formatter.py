from __future__ import annotations

from pipeline.context_builder import filter_citation_sources


def format_citations(sources: list[dict]) -> str:
    filtered_sources = filter_citation_sources(sources)
    if not filtered_sources:
        return ""

    lines = ["\n\n---", "**Sources**"]

    for idx, source in enumerate(filtered_sources[:3], start=1):
        doc = str(source.get("doc") or "policy")
        snippet = str(source.get("snippet") or "").strip().replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:120]
        if snippet and not snippet.endswith("..."):
            snippet += "..."

        lines.append(f"> [{idx}] {doc} — \"{snippet}\"")

    return "\n".join(lines)


def format_inline_citations(answer: str, sources: list[dict]) -> str:
    return str(answer) + format_citations(sources)
