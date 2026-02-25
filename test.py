"""
Balanced RAG Pipeline Test (Production-Style)

What this tests:
- Cached FAISS Index loading (Smart Caching)
- Balanced Hybrid Retrieval (Repo-Agnostic)
- Cross-Encoder Reranking
- Exact file + line tracing (Codebase Assistant ready)

Works for:
- Your repo
- Any random GitHub repo (future goal)
"""

from app.ingestion.pipeline import IngestionPipeline
from app.vectorstore.index_manager import IndexManager
from app.rag.pipeline import CodebaseRAGPipeline


class BalancedRAGTester:
    def __init__(self, repo_url: str, repo_name: str):
        self.repo_url = repo_url
        self.repo_name = repo_name

        print("\n[TEST] Initializing Balanced RAG Tester...")

        # Step 1: Ingestion pipeline (repo → chunks)
        self.ingestion = IngestionPipeline()

        # Step 2: Index manager (smart cache: disk + memory)
        self.index_manager = IndexManager()

        # Will be set later
        self.vector_store = None
        self.rag_pipeline = None

    def setup(self):
        """
        Full setup:
        - Clone repo (if needed)
        - Chunk documents
        - Load or build index (SMART CACHING)
        - Initialize RAG pipeline
        """
        print("\n[TEST] Starting setup...")

        # Step 1: Ingest + Chunk
        chunks = self.ingestion.ingest_from_github(self.repo_url)
        print(f"[TEST] Total chunks generated: {len(chunks)}")

        # Step 2: Load or Build Cached Index (VERY IMPORTANT)
        self.vector_store = self.index_manager.load_or_build_index(
            repo_name=self.repo_name,
            chunks=chunks
        )

        # Step 3: Initialize Balanced RAG Pipeline
        self.rag_pipeline = CodebaseRAGPipeline(
            vector_store=self.vector_store,
            embedder=self.index_manager.embedder
        )

        print("[TEST] Setup completed successfully.")

    def query(self, user_query: str, top_k: int = 3):
        """
        Run a query through the full pipeline.
        """
        if not self.rag_pipeline:
            raise Exception("Pipeline not initialized. Call setup() first.")

        print(f"\n🧠 USER QUERY: {user_query}")

        # Step 4: Get final context (retrieval + reranking)
        results = self.rag_pipeline.get_context(
            query=user_query,
            final_top_k=top_k
        )

        # Step 5: Pretty Print (Exact Location + Debug Info)
        self._print_results(results)

        return results

    def _print_results(self, results):
        """
        Developer-friendly output:
        - Exact file
        - Exact line numbers
        - Chunk type
        - Scores
        - Code preview
        """
        print("\n🔎 FINAL CONTEXT RESULTS (Copilot-Style):\n")

        if not results:
            print("❌ No relevant chunks found.")
            return

        for i, res in enumerate(results, 1):
            meta = res["metadata"]

            file_name = meta.get("file_name", "unknown")
            file_path = meta.get("file_path", "unknown")
            language = meta.get("language", "unknown")
            chunk_type = meta.get("chunk_type", "unknown")
            start_line = meta.get("start_line", "N/A")
            end_line = meta.get("end_line", "N/A")
            rerank_score = res.get("rerank_score", 0.0)
            hybrid_score = res.get("hybrid_score", 0.0)

            preview = res["content"][:200].replace("\n", " ")

            print(f"📌 Result {i}")
            print(f"📄 File: {file_name}")
            print(f"📂 Full Path: {file_path}")
            print(f"💻 Language: {language}")
            print(f"🧩 Chunk Type: {chunk_type}")
            print(f"📍 Lines: {start_line} → {end_line}")
            print(f"⭐ Hybrid Score: {hybrid_score:.4f}")
            print(f"🏆 Rerank Score: {rerank_score:.4f}")
            print(f"🔎 Preview: {preview}...")
            print("=" * 80)


# -------------------------
# MAIN RUNNER (Reusable)
# -------------------------
if __name__ == "__main__":
    # 🔹 You can change this to ANY repo later (flexible)
    repo_url = "https://github.com/sai-rohan-bantu/SystemDesign-LLD-Behavioural"
    repo_name = "SystemDesign-LLD-Behavioural"

    tester = BalancedRAGTester(repo_url, repo_name)

    # One-time setup (index will be cached after first run)
    tester.setup()

    # Test Queries (Copilot-style)
    tester.query("Explain strategy pattern implementation in this repo", top_k=3)
    tester.query("Where is iterator implementation?", top_k=3)
    tester.query("Explain project architecture", top_k=3)