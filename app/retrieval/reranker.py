"""
Cross Encoder Reranker (Post-Index Stage)

Improves precision after vector retrieval.
CPU-optimized for 8GB RAM systems.
"""

from typing import List, Dict
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print("[RERANKER] Loading lightweight cross-encoder...")
        self.model = CrossEncoder(model_name)
        print("[RERANKER] Ready.")

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Production reranking with:
        - Cross-encoder scoring
        - File diversity filtering (VERY IMPORTANT for codebases)
        """

        if not candidates:
            return []

        # Prepare (query, chunk_text) pairs
        pairs = [(query, c["content"]) for c in candidates]

        # Predict relevance scores
        scores = self.model.predict(pairs)

        # Attach scores to each chunk
        for chunk, score in zip(candidates, scores):
            chunk["rerank_score"] = float(score)

        # Sort by rerank score (descending)
        reranked = sorted(
            candidates,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        # 🔥 File diversity filtering (industry practice for codebases)
        unique_results = []
        seen_files = set()

        for chunk in reranked:
            file_path = chunk["metadata"].get("file_path")

            if file_path not in seen_files:
                unique_results.append(chunk)
                seen_files.add(file_path)

            if len(unique_results) >= top_k:
                break

        print(f"[RERANKER] Reranked {len(reranked)} chunks → {len(unique_results)} unique files.")
        return unique_results