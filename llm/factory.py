from llm.ollama_llm import OllamaLLM
from llm.openrouter_llm import OpenRouterLLM

def get_llm(provider: str):
    if provider == "GPT-4o Mini(OpenRouter)":
        return OpenRouterLLM(model="openai/gpt-4o-mini")

    elif provider == "Claude Haiku(OpenRouter)":
        return OpenRouterLLM(model="anthropic/claude-3-haiku")

    elif provider == "Mistral(Ollama)":
        return OllamaLLM(model="mistral:7b")

    else:
        raise ValueError(f"Unknown provider: {provider}")
