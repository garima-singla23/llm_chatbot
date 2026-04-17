from __future__ import annotations


DOC_TYPE_KEYWORDS = {
    "baggage": [
        "baggage",
        "luggage",
        "suitcase",
        "bag",
        "carry",
        "hand baggage",
        "checked",
        "allowance",
        "kg",
        "weight",
    ],
    "cancellation": [
        "cancel",
        "refund",
        "reschedule",
        "change date",
        "no-show",
        "cancellation fee",
        "money back",
    ],
    "checkin": [
        "check in",
        "check-in",
        "checkin",
        "boarding",
        "gate",
        "boarding pass",
        "seat",
        "web check",
    ],
}


def detect_doc_type(query: str) -> str | None:
    text = (query or "").lower()
    if not text:
        return None

    matches: dict[str, int] = {}
    for doc_type, keywords in DOC_TYPE_KEYWORDS.items():
        matches[doc_type] = sum(1 for keyword in keywords if keyword in text)

    top_count = max(matches.values(), default=0)
    if top_count < 2:
        return None

    winners = [doc_type for doc_type, count in matches.items() if count == top_count]
    if len(winners) != 1:
        return None

    return winners[0]


class MetadataFilter:
    def __init__(self, chunk_metadata: list[dict]):
        self.metadata = chunk_metadata

    def get_ids_for_type(self, doc_type: str) -> list[int]:
        return [m["chunk_id"] for m in self.metadata if m.get("doc_type") == doc_type]

    def filter_results(self, chunk_ids: list[int], doc_type: str | None) -> list[int]:
        if doc_type is None:
            return chunk_ids

        allowed = set(self.get_ids_for_type(doc_type))
        filtered = [chunk_id for chunk_id in chunk_ids if chunk_id in allowed]
        if not filtered:
            return chunk_ids

        return filtered