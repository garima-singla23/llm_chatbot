from llm.ollama_llm import OllamaLLM
from llm.openai_llm import OPENAILLM
from llm.openrouter_llm import OpenRouterLLM

def get_llm(provider: str):
    normalized = provider.lower().replace(" ", "")

    if normalized == "gpt-4omini(openrouter)":
        return OpenRouterLLM(model="openai/gpt-4o-mini")

    elif normalized == "claudehaiku(openrouter)":
        return OpenRouterLLM(model="anthropic/claude-3-haiku")

    elif "gpt" in normalized:
        return OPENAILLM(model=provider.strip())

    elif normalized == "phi-3(ollama)" or normalized == "phi3(ollama)":
        return OllamaLLM(model="phi3:latest")

    elif normalized == "mistral(ollama)":
        return OllamaLLM(model="mistral:7b")

    else:
        raise ValueError(f"Unknown provider: {provider}")
