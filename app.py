import gradio as gr
from llm.factory import get_llm

with open("prompts/system.txt") as f:
    system_prompt = f.read()

def reset_chat():
    return [], []

def stream_chat(model_choice, user_input, history):
    llm = get_llm(model_choice)

    messages = [{"role": "system", "content": system_prompt}]
    for u, a in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})

    messages.append({"role": "user", "content": user_input})

    stream = llm.chat(messages, stream=True)

    partial = ""
    base_history = history.copy()

    for token in stream:
        partial += token
        updated_history = base_history + [(user_input, partial)]

        yield updated_history, "", updated_history
        
with gr.Blocks() as ui:
    gr.Markdown("## ✈️ AI Airline Customer Support Assistant")

    model_choice = gr.Dropdown(
        ["GPT-4o Mini (OpenRouter)", "Mistral(Ollama)"],
        value="Mistral(Ollama)",
        label="Select Model"
    )

    chatbot = gr.Chatbot(label="conversation")
    history_state = gr.State([])  
    user_input = gr.Textbox(placeholder="Ask your airline-related question...")
    send = gr.Button("Send")
    reset = gr.Button("Reset Chat")

    send.click(
        fn=stream_chat,
        inputs=[model_choice, user_input, history_state],
        outputs=[chatbot,user_input,history_state]
    )
    user_input.submit(
    fn=stream_chat,
    inputs=[model_choice, user_input, history_state],
    outputs=[chatbot, user_input, history_state]
)
    reset.click(
    fn=reset_chat,
    inputs=[],
    outputs=[chatbot, history_state]
)

ui.queue().launch()
