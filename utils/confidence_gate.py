from __future__ import annotations

RETRIEVAL_THRESHOLD = 0.35
HYBRID_THRESHOLD = 0.40


def should_fallback(
    sources: list[dict],
    threshold: float = None,
) -> tuple[bool, float]:
    if threshold is None:
        threshold = RETRIEVAL_THRESHOLD

    if not sources:
        return True, 0.0

    scores = []
    for source in sources:
        try:
            scores.append(float(source.get("score", 0.0)))
        except (TypeError, ValueError, AttributeError):
            scores.append(0.0)

    max_score = max(scores) if scores else 0.0

    if max_score < threshold:
        return True, max_score

    return False, max_score


def build_fallback_response(query: str, max_score: float) -> str:
    return (
        "I don't have specific policy information to confidently answer "
        f"'{query[:80]}'.\n\n"
        "For accurate information, please check:\n"
        "- The airline's official website\n"
        "- Call the airline's customer support\n"
        "- Check your booking confirmation email\n\n"
        f"(Retrieval confidence: {max_score:.0%})"
    )
