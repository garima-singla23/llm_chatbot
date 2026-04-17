# debug_chat.py — run this to manually inspect any query
from gradio_client import Client

client = Client("http://127.0.0.1:7860")

queries = [
    "cheapest flights from Delhi to Mumbai next Friday",
    "what is the baggage allowance?",
    "compare IndiGo vs Air India BLR to DXB",
    "flights to London",           # missing origin
    "what about business class?",  # follow-up with no history
]

for q in queries:
    print(f"\n{'='*60}")
    print(f"QUERY: {q}")
    print(f"{'='*60}")
    result = client.predict(message=q, history=[], api_name="/chat")
    print(result)
    print()