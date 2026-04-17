import os
import time

from utils.flightapi_parser import debug_print_raw, parse_flightapi_response


def test_flightapi_key_loaded():
    key = os.getenv("FLIGHTAPI_KEY")
    assert key is not None
    assert len(key) > 10


def test_flightapi_search_returns_data(flightapi_client):
    result = flightapi_client.search_flights("DEL", "BOM", "2026-06-01")
    assert result is not None
    assert isinstance(result, (list, dict))
    assert result != []


def test_flightapi_response_parseable(flightapi_client):
    result = flightapi_client.search_flights("DEL", "BOM", "2026-06-01")
    offers = parse_flightapi_response(result)
    assert len(offers) > 0
    assert offers[0].total_price > 0
    assert offers[0].currency in ("INR", "USD")
    assert len(offers[0].segments) > 0


def test_flightapi_invalid_route_returns_empty(flightapi_client):
    result = flightapi_client.search_flights("ZZZ", "ZZZ", "2026-06-01")
    assert result == [] or isinstance(result, dict)


def test_flightapi_cache_works(flightapi_client):
    first = flightapi_client.search_flights("DEL", "BOM", "2026-06-01")

    start = time.perf_counter()
    second = flightapi_client.search_flights("DEL", "BOM", "2026-06-01")
    elapsed = time.perf_counter() - start

    assert elapsed < 0.05
    assert first == second


def test_flightapi_offer_fields(flightapi_client):
    result = flightapi_client.search_flights("DEL", "BOM", "2026-06-01")
    offers = parse_flightapi_response(result)

    assert len(offers) > 0

    for offer in offers:
        assert offer.offer_id is not None
        assert offer.total_price > 0
        assert offer.num_stops >= 0
        assert offer.validating_airline is not None or offer.validating_airline == ""
        assert offer.total_duration_minutes > 0


def test_debug_print_does_not_crash(flightapi_client):
    result = flightapi_client.search_flights("DEL", "BOM", "2026-06-01")
    debug_print_raw(result)
    assert True
