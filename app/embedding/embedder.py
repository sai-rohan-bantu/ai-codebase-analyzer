"""
Local Embedding Module (Production + CPU Optimized)

Purpose:
- Convert semantic code chunks into vector embeddings
- Fully offline (zero cost)
- Optimized for 8GB RAM laptops
- Supports multi-language code (Java, Python, JS, HTML, MD)
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


class LocalEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Lightweight embedding model.
        Best choice for:
        - CPU inference
        - Low RAM systems
        - Code + text semantic search
        """
        print("[EMBEDDER] Loading embedding model (this happens once)...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"[EMBEDDER] Model loaded. Embedding dimension: {self.dimension}")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Convert list of chunk texts into embeddings.

        Used during:
        - Index building phase
        - Bulk embedding (repo chunks)
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            batch_size=16,              # Safe for 8GB RAM
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Improves cosine similarity search
        )

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Convert user query into embedding vector.

        Used during:
        - Retrieval phase (search)
        - Agent queries like:
          "Explain strategy pattern"
          "Where is payment logic?"
        """
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding[0]