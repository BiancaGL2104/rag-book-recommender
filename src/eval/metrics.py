def recall_at_k(retrieved_titles, relevant_titles, k):
    retrieved_top_k = retrieved_titles[:k]
    hits = sum(1 for r in relevant_titles if r in retrieved_top_k)
    return hits / len(relevant_titles) if relevant_titles else 0.0

def precision_at_k(retrieved_titles, relevant_titles, k):
    retrieved_top_k = retrieved_titles[:k]
    hits = sum(1 for r in relevant_titles if r in retrieved_top_k)
    return hits / k

def reciprocal_rank(retrieved_titles, relevant_titles):
    for i, title in enumerate(retrieved_titles):
        if title in relevant_titles:
            return 1 / (i + 1)
    return 0.0
