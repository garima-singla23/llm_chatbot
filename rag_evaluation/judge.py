def llm_judge(judge_model, question, context, answer):

    prompt = f"""
    Evaluate the RAG answer.

    Question: {question}
    Context: {context}
    Answer: {answer}

    Score from 1-10:
    - Faithfulness
    - Relevance
    - Hallucination Risk

    Return JSON.
    """

    response = judge_model.invoke(prompt)
    return response
