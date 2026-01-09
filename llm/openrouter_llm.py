import requests
from llm.base import BaseLLM

class OpenRouterLLM(BaseLLM):
    def __init__(self, model):
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def chat(self, messages,stream=False):
        headers = {
            "Authorization": f"Bearer {self._get_api_key()}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "AI Airline Assistant"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "stream":stream,
            "temperature":0.3
        }

        response = requests.post(self.url, headers=headers, json=payload)
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]

    def _get_api_key(self):
        import os
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY not set")
        return key
