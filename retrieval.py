
"""Hybrid retrieval combining dense semantic search and BM25."""

import re

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def tokenize(text):
    """Tokenize text for BM25 keyword retrieval."""
    return re.findall(r"\b\w+\b", text.lower())


def split_text(text, chunk_size=1000, overlap=150):
    """Split text into overlapping chunks."""
    text = " ".join(text.split())

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


class HybridRetriever:
    """Retrieve passages using dense search and BM25."""

    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunks = []
        self.bm25 = None
        self.tokenized_chunks = []

    def fit(self, chunks):
        """Build the dense index and BM25 index."""
        self.chunks = list(chunks)

        if not self.chunks:
            self.model = None
            self.index = None
            self.bm25 = None
            self.tokenized_chunks = []
            return self

        if any(not isinstance(chunk, str) or not chunk.strip()
               for chunk in self.chunks):
            raise ValueError("All chunks must be non-empty strings")

        self.tokenized_chunks = [tokenize(chunk) for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks)

        self.model = SentenceTransformer(self.model_name)
        embeddings = self.model.encode(
            self.chunks,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype="float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        return self

    def search(self, query, top_k=5, semantic_weight=0.5):
        """Combine semantic and keyword rankings with reciprocal rank fusion."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")

        if not 0.0 <= semantic_weight <= 1.0:
            raise ValueError("semantic_weight must be between 0 and 1")

        if not self.chunks:
            return []

        candidate_count = min(len(self.chunks), max(top_k * 3, 10))

        # Dense semantic retrieval
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_embedding = np.asarray(query_embedding, dtype="float32")
        dense_scores, dense_indices = self.index.search(
            query_embedding, candidate_count
        )

        dense_rank = {
            int(idx): rank
            for rank, idx in enumerate(dense_indices[0], start=1)
            if idx >= 0
        }

        # BM25 keyword retrieval
        keyword_scores = self.bm25.get_scores(tokenize(query))
        keyword_indices = np.argsort(keyword_scores)[::-1][:candidate_count]

        keyword_rank = {
            int(idx): rank
            for rank, idx in enumerate(keyword_indices, start=1)
        }

        # Reciprocal Rank Fusion (RRF)
        rrf_k = 60
        combined_indices = set(dense_rank) | set(keyword_rank)
        results = []

        for idx in combined_indices:
            dense_position = dense_rank.get(idx)
            keyword_position = keyword_rank.get(idx)

            dense_rrf = (
                1.0 / (rrf_k + dense_position)
                if dense_position is not None else 0.0
            )
            keyword_rrf = (
                1.0 / (rrf_k + keyword_position)
                if keyword_position is not None else 0.0
            )

            fused_score = (
                semantic_weight * dense_rrf
                + (1.0 - semantic_weight) * keyword_rrf
            )

            results.append({
                "text": self.chunks[idx],
                "score": float(fused_score),
                "dense_rank": dense_position,
                "keyword_rank": keyword_position,
                "chunk_index": idx,
            })

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]
