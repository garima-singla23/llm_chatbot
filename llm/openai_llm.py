import os

from openai import OpenAI
from llm.base import BaseLLM

class OPENAILLM(BaseLLM):
    def __init__(self, model="gpt-4o-mini", temperature=0.3):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def chat(self, messages, stream=False):
        if stream:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=500,
                stream=True,
            )

            def generate():
                for chunk in response:
                    delta = chunk.choices[0].delta
                    content = delta.content if delta else None
                    if content:
                        yield content

            return generate()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=500
        )
        return response.choices[0].message.content

