import os
import sys
import gradio as gr
from llm.factory import get_llm
from rag.retriever import retrieve_docs
from pipeline.router import route
from apis.flightapi_client import FlightAPIClient
from apis.aviationstack_client import AviationstackClient

# Initialize flight search client
flightapi_key = os.getenv("FLIGHTAPI_KEY")
if not flightapi_key:
    print(
        "[WARNING] FLIGHTAPI_KEY not set — flight search will return empty results",
        file=sys.stderr,
    )

flight_client = FlightAPIClient(api_key=flightapi_key or "")

aviationstack_client = AviationstackClient(
    api_key=os.getenv("AVIATIONSTACK_API_KEY", "")
)

show_dashboard = os.getenv("SHOW_DASHBOARD", "false").lower() == "true"

# Persist entity extraction across turns during clarification flows
pending_entities = {}

def reset_chat():
    global pending_entities
    pending_entities = {}
    return [], []

def stream_chat(model_choice, user_input, history):
    global pending_entities
    base_history = history.copy()

    try:
        llm = get_llm(model_choice)
    except Exception as exc:
        error_message = f"I could not initialize {model_choice}: {exc}"
        updated_history = base_history + [(user_input, error_message)]
        yield updated_history, "", updated_history
        return

    # Convert Gradio chat history format to list of tuples for router
    # Gradio history is [(user, assistant), ...]
    chat_history = [(u, a) for u, a in base_history]

    try:
        # Use router to handle intent classification and orchestration
        response, updated_pending = route(
            query=user_input,
            llm=llm,
            retriever=retrieve_docs,
            flight_client=flight_client,
            chat_history=chat_history,
            pending_entities=pending_entities,
            aviationstack_client=aviationstack_client,
        )

        # Update pending entities for next turn (for clarification flows)
        pending_entities = updated_pending.copy() if updated_pending else {}

        # Stream response character by character
        partial = ""
        for char in response:
            partial += char
            updated_history = base_history + [(user_input, partial)]
            yield updated_history, "", updated_history

    except Exception as exc:
        error_message = f"I encountered an error: {exc}"
        updated_history = base_history + [(user_input, error_message)]
        yield updated_history, "", updated_history

        
with gr.Blocks() as ui:
    gr.Markdown("## AI Airline Customer Support Assistant")

    with gr.Tabs():
        with gr.Tab("Chat"):
            model_choice = gr.Dropdown(
                ["GPT-4o Mini (OpenRouter)", "Claude Haiku (OpenRouter)", "Phi-3 (Ollama)", "Mistral (Ollama)", "gpt-4o", "gpt-3.5-turbo"],
                value="GPT-4o Mini (OpenRouter)",
                label="Select Model"
            )

            chatbot = gr.Chatbot(label="conversation", render_markdown=True)
            history_state = gr.State([])
            user_input = gr.Textbox(placeholder="Ask your airline-related question...")
            send = gr.Button("Send")
            reset = gr.Button("Reset Chat")

            send.click(
                fn=stream_chat,
                inputs=[model_choice, user_input, history_state],
                outputs=[chatbot, user_input, history_state]
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

        if show_dashboard:
            with gr.Tab("Eval"):
                gr.Markdown(
                    "Open the evaluation dashboard at: "
                    "[http://127.0.0.1:7861](http://127.0.0.1:7861)"
                )
                gr.HTML(
                    '<a href="http://127.0.0.1:7861" target="_blank">'
                    '<button style="padding:8px 14px;">Open Eval Dashboard</button>'
                    '</a>'
                )

if __name__ == "__main__":
    ui.queue().launch()
