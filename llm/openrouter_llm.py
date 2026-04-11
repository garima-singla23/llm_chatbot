import requests
from llm.base import BaseLLM
import json

class OpenRouterLLM(BaseLLM):
    def __init__(self, model):
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def chat(self, messages, stream=False):
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

        response = requests.post(self.url, headers=headers, json=payload, stream=stream, timeout=60)
        response.raise_for_status()

        if not stream:
            return response.json()["choices"][0]["message"]["content"]

        def generate():
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue

                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue

                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break

                chunk = json.loads(data)
                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content

        return generate()

    def _get_api_key(self):
        import os
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY not set")
        return key
