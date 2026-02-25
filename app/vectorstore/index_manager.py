"""
Smart Index Manager (Production-Grade)

Features:
- Auto build index if not exists
- Auto load from disk if exists
- Avoid re-embedding (huge performance gain)
- Designed for codebase RAG systems
"""

import os
from app.embedding.embedder import LocalEmbedder
from app.vectorstore.faiss_store import FAISSVectorStore


class IndexManager:
    def __init__(self, index_dir: str = "data/index"):
        """
        index_dir: root folder for all indexes (per repo)
        """
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)

        self.embedder = LocalEmbedder()

    def get_index_path(self, repo_name: str) -> str:
        """
        Creates per-repo index path (industry standard).
        """
        safe_name = repo_name.replace("/", "_")
        return os.path.join(self.index_dir, f"{safe_name}.faiss")

    def load_or_build_index(self, repo_name: str, chunks: list):
        """
        Smart Caching Logic:
        1. If index exists → Load (fast)
        2. Else → Build + Save (one-time cost)
        """
        index_path = self.get_index_path(repo_name)

        # Initialize vector store
        vector_store = FAISSVectorStore(
            dimension=self.embedder.dimension,
            index_path=index_path,
        )

        # Try loading existing index
        if vector_store.load():
            print("[INDEX MANAGER] Using cached index (fast startup).")
            return vector_store

        print("[INDEX MANAGER] No index found. Building new index...")

        # Prepare data for embedding
        texts = [chunk["content"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        # Generate embeddings (heavy step, only once)
        embeddings = self.embedder.embed_texts(texts)

        # Add to vector store
        vector_store.add(embeddings, texts, metadatas)

        # Save for future runs (Smart Cache)
        vector_store.save()

        print("[INDEX MANAGER] Index built and cached successfully.")
        return vector_store