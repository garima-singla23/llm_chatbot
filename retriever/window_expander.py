from __future__ import annotations


def _expanded_indices(chunk_ids: list[int], total: int, window: int) -> list[int]:
    if total <= 0:
        return []

    seen: set[int] = set()
    expanded: list[int] = []

    for chunk_id in chunk_ids:
        if not isinstance(chunk_id, int):
            continue

        start = max(0, chunk_id - window)
        end = min(total - 1, chunk_id + window)

        for idx in range(start, end + 1):
            if idx not in seen:
                seen.add(idx)
                expanded.append(idx)

    # Keep final output in original document order.
    expanded.sort()
    return expanded


def expand_window(
    chunk_ids: list[int],
    all_chunks: list[str],
    window: int = 1,
) -> list[str]:
    indices = _expanded_indices(chunk_ids, len(all_chunks), window)
    return [all_chunks[idx] for idx in indices]


def expand_with_metadata(
    sources: list[dict],
    all_chunks: list[str],
    window: int = 1,
) -> tuple[list[str], list[dict]]:
    if not sources:
        return [], []

    top_sources: list[tuple[int, dict]] = []
    for source in sources:
        try:
            chunk_id = int(source.get("chunk_id"))
        except (TypeError, ValueError, AttributeError):
            continue
        if 0 <= chunk_id < len(all_chunks):
            top_sources.append((chunk_id, source))

    if not top_sources:
        return [], []

    top_chunk_ids = [chunk_id for chunk_id, _ in top_sources]
    expanded_indices = _expanded_indices(top_chunk_ids, len(all_chunks), window)

    top_source_by_id = {chunk_id: src for chunk_id, src in top_sources}

    expanded_texts: list[str] = []
    expanded_sources: list[dict] = []

    for idx in expanded_indices:
        chunk_text = all_chunks[idx]
        expanded_texts.append(chunk_text)

        if idx in top_source_by_id:
            source = dict(top_source_by_id[idx])
            source["chunk_id"] = idx
            source["snippet"] = str(source.get("snippet") or chunk_text[:120])
            source["expanded"] = False
            source["score"] = float(source.get("score", 0.0))
            expanded_sources.append(source)
            continue

        influencing_scores = []
        for top_idx, src in top_sources:
            if max(0, top_idx - window) <= idx <= min(len(all_chunks) - 1, top_idx + window):
                try:
                    influencing_scores.append(float(src.get("score", 0.0)))
                except (TypeError, ValueError, AttributeError):
                    influencing_scores.append(0.0)

        base_score = max(influencing_scores) if influencing_scores else 0.0
        expanded_sources.append(
            {
                "doc": str(top_sources[0][1].get("doc") or "policy"),
                "chunk_id": idx,
                "score": base_score * 0.7,
                "snippet": chunk_text[:120],
                "expanded": True,
            }
        )

    return expanded_texts, expanded_sources
