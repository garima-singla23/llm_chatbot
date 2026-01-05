def build_messages(system_prompt: str, user_input: str):
    """
    Builds messages in OpenAI / Ollama compatible format.

    Args:
        system_prompt (str): Instructions for the assistant
        user_input (str): User's question

    Returns:
        list[dict]: Messages formatted for LLM chat APIs
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
