import requests
from llm.base import BaseLLM
import json

class OllamaLLM(BaseLLM):
    def __init__(self, model):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def chat(self, messages,stream=True):
        # Convert chat messages into a single prompt
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt += f"{role.upper()}: {content}\n"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options":{"num_predict":200}

        }

        response = requests.post(self.url, json=payload, stream=True)
        response.raise_for_status()

        if not stream:
            return response.json()["response"]
        
        def generate():
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line.decode("utf-8"))
                if "response" in chunk:
                    yield chunk["response"]
        return generate()

