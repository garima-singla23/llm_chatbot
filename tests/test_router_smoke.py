import unittest

import pipeline.router as router


class _DummyLLM:
    def chat(self, messages, stream=False):
        return "ok"


class _DummyFlightClient:
    def search_flights(self, origin, destination, date, cabin, adults):
        return []


class _DummyAviationstackClient:
    def get_flight_status(self, flight_number):
        return {
            "flight_number": flight_number,
            "airline": "AI",
            "status": "active",
            "departure_airport": "DEL",
            "arrival_airport": "BOM",
            "scheduled_departure": "2026-04-11T10:00:00+05:30",
            "actual_departure": "2026-04-11T10:10:00+05:30",
            "scheduled_arrival": "2026-04-11T12:00:00+05:30",
            "actual_arrival": None,
            "delay_minutes": 10,
        }


class RouterSmokeTests(unittest.TestCase):
    def test_policy_route_accepts_callable_retriever(self):
        llm = _DummyLLM()

        def retriever_fn(query):
            return ["Policy chunk"]

        result, pending = router.route(
            query="What is your baggage policy?",
            llm=llm,
            retriever=retriever_fn,
            flight_client=_DummyFlightClient(),
            chat_history=[],
            pending_entities={},
        )

        self.assertIsInstance(result, str)
        self.assertEqual(pending, {})
        self.assertNotIn("unexpected error", result.lower())

    def test_flight_status_route_uses_provided_client(self):
        original_classify_intent = router.classify_intent
        try:
            router.classify_intent = lambda query, llm, chat_history: {
                "type": "flight_status",
                "confidence": 1.0,
            }

            result, pending = router.route(
                query="status for AI101",
                llm=_DummyLLM(),
                retriever=lambda q: [],
                flight_client=_DummyFlightClient(),
                chat_history=[],
                pending_entities={},
                aviationstack_client=_DummyAviationstackClient(),
            )

            self.assertIsInstance(result, str)
            self.assertEqual(pending, {})
            self.assertNotIn("not currently available", result.lower())
        finally:
            router.classify_intent = original_classify_intent


if __name__ == "__main__":
    unittest.main()
