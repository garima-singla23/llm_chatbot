from __future__ import annotations

from utils.entity_extractor import extract_entities


FLIGHT_EVAL_QUERIES = [
    {
        "query": "cheapest flights from Delhi to Mumbai next Friday economy",
        "expected": {
            "origin_iata": "DEL",
            "destination_iata": "BOM",
            "cabin_class": "ECONOMY",
            "adults": 1,
        },
    },
    {
        "query": "IndiGo business class BLR to DXB April 20",
        "expected": {
            "origin_iata": "BLR",
            "destination_iata": "DXB",
            "cabin_class": "BUSINESS",
            "preferred_airline_iata": "6E",
        },
    },
    {
        "query": "one way Bangalore to Singapore 2 adults tomorrow",
        "expected": {
            "origin_iata": "BLR",
            "destination_iata": "SIN",
            "adults": 2,
        },
    },
    {
        "query": "flights from Delhi to London next Monday",
        "expected": {
            "origin_iata": "DEL",
            "destination_iata": "LHR",
        },
    },
    {
        "query": "Air India economy New Delhi to Dubai this weekend",
        "expected": {
            "origin_iata": "DEL",
            "destination_iata": "DXB",
            "cabin_class": "ECONOMY",
            "preferred_airline_iata": "AI",
        },
    },
    {
        "query": "BOM to SIN April 15 2026 business",
        "expected": {
            "origin_iata": "BOM",
            "destination_iata": "SIN",
            "cabin_class": "BUSINESS",
        },
    },
    {
        "query": "3 adults from Hyderabad to Goa tomorrow",
        "expected": {
            "origin_iata": "HYD",
            "destination_iata": "GOI",
            "adults": 3,
        },
    },
    {
        "query": "cheapest flight to Goa from Mumbai",
        "expected": {
            "origin_iata": "BOM",
            "destination_iata": "GOI",
        },
    },
    {
        "query": "flights to London",
        "expected": {
            "origin_iata": None,
            "destination_iata": "LHR",
        },
    },
    {
        "query": "SpiceJet Kolkata to Chennai economy",
        "expected": {
            "origin_iata": "CCU",
            "destination_iata": "MAA",
            "preferred_airline_iata": "SG",
        },
    },
    {
        "query": "round trip Delhi to Dubai leaving April 20 returning April 27",
        "expected": {
            "origin_iata": "DEL",
            "destination_iata": "DXB",
            "return_date": "2026-04-27",
        },
    },
    {
        "query": "premium economy from Mumbai to London next month",
        "expected": {
            "origin_iata": "BOM",
            "destination_iata": "LHR",
            "cabin_class": "PREMIUM_ECONOMY",
        },
    },
    {
        "query": "first class BLR to JFK 1 adult",
        "expected": {
            "origin_iata": "BLR",
            "destination_iata": "JFK",
            "cabin_class": "FIRST",
            "adults": 1,
        },
    },
    {
        "query": "Bombay to Calcutta business",
        "expected": {
            "origin_iata": "BOM",
            "destination_iata": "CCU",
            "cabin_class": "BUSINESS",
        },
    },
    {
        "query": "Madras to Delhi tomorrow",
        "expected": {
            "origin_iata": "MAA",
            "destination_iata": "DEL",
        },
    },
    {
        "query": "DEL to DXB economy 2 children 2 adults",
        "expected": {
            "origin_iata": "DEL",
            "destination_iata": "DXB",
            "adults": 2,
            "children": 2,
        },
    },
    {
        "query": "go to singapore next month",
        "expected": {
            "origin_iata": None,
            "destination_iata": "SIN",
        },
    },
    {
        "query": "from Bangalore flights",
        "expected": {
            "origin_iata": "BLR",
            "destination_iata": None,
        },
    },
    {
        "query": "Chennai to Kochi 15/05/2026",
        "expected": {
            "origin_iata": "MAA",
            "destination_iata": "COK",
        },
    },
    {
        "query": "AI from DEL to BOM tomorrow",
        "expected": {
            "origin_iata": "DEL",
            "destination_iata": "BOM",
            "preferred_airline_iata": "AI",
        },
    },
]


class FlightEvalHarness:
    def __init__(self, llm):
        self.llm = llm
        self.results = []

    def run(self) -> dict:
        self.results = []
        for case in FLIGHT_EVAL_QUERIES:
            extracted = extract_entities(case["query"], self.llm)
            result = self._score(extracted, case["expected"], case["query"])
            self.results.append(result)
        return self.summary()

    def _score(self, extracted: dict, expected: dict, query: str) -> dict:
        total_expected_fields = len(expected)
        matched_fields = 0
        mismatches = []

        for key, expected_value in expected.items():
            match = extracted.get(key) == expected_value
            if match:
                matched_fields += 1
            else:
                mismatches.append(key)

        field_accuracy = (
            matched_fields / total_expected_fields if total_expected_fields > 0 else 0.0
        )
        full_match = matched_fields == total_expected_fields

        return {
            "query": query,
            "field_accuracy": float(field_accuracy),
            "full_match": bool(full_match),
            "mismatches": mismatches,
        }

    def summary(self) -> dict:
        if not self.results:
            return {
                "full_match_rate": 0.0,
                "mean_field_accuracy": 0.0,
                "n_total": 0,
                "n_perfect": 0,
                "worst_cases": [],
            }

        n_total = len(self.results)
        n_perfect = sum(1 for result in self.results if result["full_match"])
        full_match_rate = n_perfect / n_total
        mean_field_accuracy = (
            sum(result["field_accuracy"] for result in self.results) / n_total
        )

        worst_cases = sorted(
            [result for result in self.results if not result["full_match"]],
            key=lambda result: result["field_accuracy"],
        )

        return {
            "full_match_rate": float(full_match_rate),
            "mean_field_accuracy": float(mean_field_accuracy),
            "n_total": n_total,
            "n_perfect": n_perfect,
            "worst_cases": worst_cases[:3],
        }
