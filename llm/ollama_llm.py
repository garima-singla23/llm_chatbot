import requests
from llm.base import BaseLLM

class OllamaLLM(BaseLLM):
    def __init__(self, model):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def chat(self, messages):
        # Convert chat messages into a single prompt
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt += f"{role.upper()}: {content}\n"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(self.url, json=payload)
        response.raise_for_status()

        data = response.json()

        if "response" not in data:
            raise ValueError(f"Unexpected Ollama response: {data}")

        return data["response"]

