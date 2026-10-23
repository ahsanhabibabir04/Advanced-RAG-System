
"""Evaluate information retrieval using standard ranking metrics."""


def reciprocal_rank(results, relevant_indices):
    """Return reciprocal rank of the first relevant result."""
    for rank, result in enumerate(results, start=1):
        if result["chunk_index"] in relevant_indices:
            return 1.0 / rank
    return 0.0


def precision_at_k(results, relevant_indices, k=5):
    """Calculate Precision@K."""
    if k < 1:
        raise ValueError("k must be at least 1")

    top_results = results[:k]

    if not top_results:
        return 0.0

    relevant_count = sum(
        result["chunk_index"] in relevant_indices
        for result in top_results
    )

    return relevant_count / k


def recall_at_k(results, relevant_indices, k=5):
    """Calculate Recall@K."""
    if k < 1:
        raise ValueError("k must be at least 1")

    if not relevant_indices:
        return 0.0

    top_results = results[:k]

    relevant_count = sum(
        result["chunk_index"] in relevant_indices
        for result in top_results
    )

    return relevant_count / len(relevant_indices)


def mean_reciprocal_rank(query_results):
    """Calculate MRR across multiple queries.

    query_results is a list of (results, relevant_indices) pairs.
    """
    if not query_results:
        return 0.0

    scores = [
        reciprocal_rank(results, relevant_indices)
        for results, relevant_indices in query_results
    ]

    return sum(scores) / len(scores)
