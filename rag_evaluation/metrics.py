def compute_metrics(evaluation_data, retriever_func, k=3):

    mrr_scores = []
    precision_scores = []
    recall_scores = []

    for item in evaluation_data:
        query = item["query"]
        ground_truth = item["ground_truth_doc"]

        docs = retriever_func(query, k)
        retrieved_ids = [
            doc.metadata.get("chunk_id")
            for doc in docs
            if doc.metadata.get("chunk_id") is not None
        ]

        rank = 0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id == ground_truth:
                rank = i + 1
                break

        if rank > 0:
            mrr_scores.append(1 / rank)
        else:
            mrr_scores.append(0)

        if ground_truth in retrieved_ids:
            precision_scores.append(1 / k)
        else:
            precision_scores.append(0)

        if ground_truth in retrieved_ids:
            recall_scores.append(1)
        else:
            recall_scores.append(0)

    return {
        "MRR": sum(mrr_scores) / len(mrr_scores),
        "Precision@K": sum(precision_scores) / len(precision_scores),
        "Recall@K": sum(recall_scores) / len(recall_scores)
    }