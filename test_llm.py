# test_llm.py
from llm.factory import get_llm

llm = get_llm("Mistral(Ollama)")
print(llm.chat([
    {"role": "user", "content": "Hello"}
]))
