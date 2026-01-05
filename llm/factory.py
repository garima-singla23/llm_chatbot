from settings import LLM_PROVIDER
from llm.openai_llm import OPENAILLM
from llm.ollama_llm import OllamaLLM

def get_llm(provider: str):
    if provider == "OpenAI":
        return OPENAILLM(model="gpt-4o-mini")
    elif provider == "Mistral(Ollama)":
        return OllamaLLM(model="mistral")
    elif provider == "LlaMA(Ollama)":
        return OllamaLLM(model="llama3.2")
    else:
        raise ValueError("unsupported LLM provider")
    
