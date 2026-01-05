import gradio as gr
from llm.factory import get_llm
from utils.formatter import build_messages

# Load system prompt
with open("prompts/system.txt") as f:
    system_prompt = f.read()

def chat(model_choice, user_input):
    try:
        llm = get_llm(model_choice)
        messages = build_messages(system_prompt, user_input)
        response = llm.chat(messages)
        return response
    except Exception as e:
        return f"ERROR: {str(e)}"

ui = gr.Interface(
    fn=chat,
    inputs=[
        gr.Dropdown(
            choices=["OpenAI", "Mistral(Ollama)", "LLaMA(Ollama)"],
            value="OpenAI",
            label="Select Model"
        ),
        gr.Textbox(
            lines=3,
            placeholder="Ask your question..."
        )
    ],
    outputs="text",
    title="Multi-Model LLM Chatbot",
    description="Select an LLM provider and ask questions dynamically"
)

ui.launch()
