from utils.confidence_gate import build_fallback_response, should_fallback


def test_fallback_triggered_on_low_scores():
    sources = [{"score": 0.2}, {"score": 0.15}]
    fallback, score = should_fallback(sources)
    assert fallback is True
    assert score == 0.2


def test_no_fallback_on_high_scores():
    sources = [{"score": 0.8}, {"score": 0.6}]
    fallback, score = should_fallback(sources)
    assert fallback is False
    assert score == 0.8


def test_fallback_on_empty_sources():
    fallback, score = should_fallback([])
    assert fallback is True
    assert score == 0.0


def test_fallback_message_contains_query():
    msg = build_fallback_response("baggage allowance economy", 0.2)
    assert "baggage allowance economy" in msg
    assert "20%" in msg
