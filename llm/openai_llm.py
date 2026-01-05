from openai import OpenAI
from llm.base import BaseLLM

class OPENAILLM(BaseLLM):
    def __init__(self,model="gpt-4o-mini",temperature=0.3):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature

    def chat(self,messages):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=500
        ) 
        return response.choices[0].message.content

