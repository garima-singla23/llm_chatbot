import time

from agents.agent_loop import run_agent
from chatbot.memory import ConversationMemory

def main():
    tests = [
        ("Check status of flight 6E204", "agentic"),
        ("What is IndiGo baggage limit", "rag"),
        ("Calculate refund for PNRTEST99", "agentic"),
    ]

    times = []

    print("\n=== LATENCY TESTS ===\n")

    for query, qtype in tests:
        memory = ConversationMemory()

        start = time.time()
        result = run_agent(query, memory, None)
        elapsed = round((time.time() - start) * 1000)

        times.append(elapsed)

        print(f"[{qtype.upper()}] {query[:40]} -> {elapsed} ms")

        result_str = str(result)
        print(f"  Response: {result_str[:80]}...")
        print()

    print("=" * 50)
    print(f"Average latency: {round(sum(times) / len(times))} ms")
    print(f"Min: {min(times)} ms")
    print(f"Max: {max(times)} ms")
    print("=" * 50)

if __name__ == "__main__":
    main()