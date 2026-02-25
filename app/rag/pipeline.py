from app.retrieval.retriever import BalancedCodeRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.rag.context_aggregator import ContextAggregator


class CodebaseRAGPipeline:
    def __init__(self, vector_store, embedder):
        self.retriever = BalancedCodeRetriever(vector_store, embedder)
        self.reranker = CrossEncoderReranker()
        self.aggregator = ContextAggregator()  # 🔥 NEW

    def get_context(self, query: str, final_top_k: int = 5):
        print("\n========== RAG PIPELINE START ==========")

        # Step 1: Hybrid Retrieval
        candidates = self.retriever.retrieve(query, top_k=25)
        print(f"[PIPELINE] Retrieved candidates: {len(candidates)}")

        # Step 2: Reranking
        reranked = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=15  # keep more for aggregation
        )

        # 🔥 Step 3: Module-aware aggregation (Copilot behavior)
        final_chunks = self.aggregator.aggregate(
            query=query,  # 🔥 PASS QUERY (CRITICAL)
            chunks=reranked,
            final_top_k=final_top_k
        )

        print(f"[PIPELINE] Final selected chunks: {len(final_chunks)}")
        print("========== RAG PIPELINE END ==========\n")

        return final_chunks