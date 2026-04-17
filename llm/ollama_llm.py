import requests
from llm.base import BaseLLM
import json

class OllamaLLM(BaseLLM):
    def __init__(self, model):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def _request(self, payload, stream, allow_fallback=True):
        try:
            response = requests.post(self.url, json=payload, stream=stream, timeout=60)
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                "Ollama is not reachable at http://localhost:11434. "
                "Start Ollama (or run 'ollama serve') and try again."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        if response.status_code < 400:
            return response

        error_text = ""
        try:
            error_text = response.json().get("error", "")
        except ValueError:
            error_text = response.text

        error_lower = error_text.lower()
        low_memory = "requires more system memory" in error_lower
        allocation_error = "unable to allocate cpu buffer" in error_lower
        runner_terminated = "llama runner process has terminated" in error_lower

        resource_failure = low_memory or allocation_error or runner_terminated

        if resource_failure and allow_fallback and payload.get("model") != "phi3:latest":
            fallback_payload = dict(payload)
            fallback_payload["model"] = "phi3:latest"
            fallback_response = self._request(fallback_payload, stream, allow_fallback=False)
            self.model = "phi3:latest"
            return fallback_response

        if resource_failure:
            raise RuntimeError(
                "Local Ollama model cannot run due to memory limits. "
                "Close other apps or switch to a cloud model in the dropdown "
                "(for example, GPT-4o Mini (OpenRouter))."
            )

        if not error_text:
            error_text = f"HTTP {response.status_code} from Ollama"
        raise RuntimeError(error_text)

    def chat(self, messages, stream=True):
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
            "options": {"num_predict": 200}

        }

        response = self._request(payload, stream)

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

