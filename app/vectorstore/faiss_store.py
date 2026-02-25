"""
FAISS Vector Store (Production + Persistent)

Responsibilities:
- Store embeddings in memory (fast search)
- Save index to disk (persistence)
- Load index from disk (smart caching)
"""

import faiss
import numpy as np
import os
import pickle


class FAISSVectorStore:
    def __init__(self, dimension: int, index_path: str):
        """
        dimension: embedding vector size (MiniLM = 384)
        index_path: where index will be stored on disk
        """
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = index_path + "_metadata.pkl"
        self.texts_path = index_path + "_texts.pkl"

        # In-memory stores (fast retrieval)
        self.index = faiss.IndexFlatIP(dimension)  # cosine similarity
        self.metadata_store = []
        self.text_store = []

    def add(self, embeddings: np.ndarray, texts: list, metadatas: list):
        """
        Add embeddings + texts + metadata to FAISS index.
        """
        if len(embeddings) == 0:
            return

        self.index.add(embeddings)
        self.text_store.extend(texts)
        self.metadata_store.extend(metadatas)

        print(f"[FAISS] Added {len(texts)} chunks to index.")

    def save(self):
        """
        Save index + metadata + texts to disk (Smart Caching).
        """
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, self.index_path)

        # Save metadata
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata_store, f)

        # Save texts
        with open(self.texts_path, "wb") as f:
            pickle.dump(self.text_store, f)

        print(f"[FAISS] Index saved to disk: {self.index_path}")

    def load(self):
        """
        Load index + metadata + texts from disk.
        """
        if not os.path.exists(self.index_path):
            return False

        self.index = faiss.read_index(self.index_path)

        with open(self.metadata_path, "rb") as f:
            self.metadata_store = pickle.load(f)

        with open(self.texts_path, "rb") as f:
            self.text_store = pickle.load(f)

        print(f"[FAISS] Index loaded from disk.")
        return True

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """
        Perform semantic search on indexed chunks.
        """
        if self.index.ntotal == 0:
            return []

        query_embedding = np.array([query_embedding])
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.text_store):
                results.append({
                    "score": float(score),
                    "content": self.text_store[idx],
                    "metadata": self.metadata_store[idx],
                })

        return results