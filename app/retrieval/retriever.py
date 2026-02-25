"""
Balanced Copilot-Style Retriever (Post-Indexing, Repo-Agnostic)

Goals:
- Works on ANY GitHub repo (no hardcoding)
- Balanced for code search + explanation
- Avoids boilerplate dominance (Main, App, Index files)
- Prefers meaningful code structures (classes, methods)
- Uses hybrid scoring (semantic + structural + keyword)
"""

from typing import List, Dict
import re


class BalancedCodeRetriever:
    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder

        # Universal boilerplate filenames (repo-agnostic)
        self.boilerplate_files = {
            "main.java", "app.java", "index.js", "main.py",
            "app.py", "index.ts", "main.ts"
        }

    # -------------------------
    # Public Entry Point
    # -------------------------
    def retrieve(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        Copilot-style balanced retrieval:
        1. High recall semantic search
        2. Hybrid scoring (structure + keyword + file importance)
        3. Return enriched candidates for reranker
        """
        print(f"\n[RETRIEVER] Query: {query}")

        # Step 1: Expand query (generic, NOT hardcoded)
        expanded_query = self._expand_query(query)

        # Step 2: Embed query
        query_embedding = self.embedder.embed_query(expanded_query)

        # Step 3: High recall retrieval (very important for reranker)
        initial_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=50  # high recall for better precision later
        )

        print(f"[RETRIEVER] Initial candidates: {len(initial_results)}")

        # Step 4: Apply hybrid scoring (repo-agnostic)
        scored_results = self._apply_hybrid_scoring(
            query=query,
            results=initial_results
        )

        # Step 5: Sort by hybrid score (before reranker)
        scored_results = sorted(
            scored_results,
            key=lambda x: x["hybrid_score"],
            reverse=True
        )

        # Return broader set for reranker
        final_candidates = scored_results[:top_k]
        print(f"[RETRIEVER] Selected candidates for reranker: {len(final_candidates)}")

        return final_candidates

    # -------------------------
    # Query Expansion (Generic)
    # -------------------------
    def _expand_query(self, query: str) -> str:
        """
        Lightweight semantic expansion (repo-agnostic).
        Improves retrieval without hardcoding domains.
        """
        query = query.strip()
        lower = query.lower()

        expansion_terms = []

        if "explain" in lower:
            expansion_terms.extend(["implementation", "logic", "class", "method"])

        if "where" in lower or "find" in lower:
            expansion_terms.extend(["definition", "source code", "function"])

        expanded = query + " " + " ".join(expansion_terms)
        return expanded

    # -------------------------
    # Hybrid Scoring (KEY PART)
    # -------------------------
    def _apply_hybrid_scoring(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Combines:
        - Semantic relevance (already from vector search)
        - Structural importance (class > method > text)
        - File importance (avoid boilerplate)
        - Keyword/path relevance (dynamic, no hardcoding)
        """
        query_keywords = self._extract_keywords(query)

        for r in results:
            meta = r["metadata"]
            file_name = meta.get("file_name", "").lower()
            file_path = meta.get("file_path", "").lower()
            chunk_type = meta.get("chunk_type", "text_chunk")

            score = 0.0

            # 1️⃣ Structural Importance (Universal for all repos)
            if chunk_type == "class_declaration":
                score += 5.0
            elif chunk_type == "method_declaration":
                score += 3.0
            else:
                score += 1.0  # text/doc chunks lowest

            # 2️⃣ Penalize Boilerplate Entry Files (Generic rule)
            if file_name in self.boilerplate_files:
                score -= 4.0  # avoid Main.java dominance

            # 3️⃣ Chunk Size Heuristic (richer context = better explanation)
            start = meta.get("start_line", 0) or 0
            end = meta.get("end_line", 0) or 0
            size = max(0, end - start)

            size_boost = min(size / 20.0, 3.0)  # capped
            score += size_boost

            # 4️⃣ Dynamic Keyword Matching (NOT hardcoded to any repo)
            keyword_hits = sum(1 for kw in query_keywords if kw in file_path)
            score += keyword_hits * 2.0

            # 5️⃣ Small boost for non-test, non-config code
            if any(ext in file_name for ext in [".java", ".py", ".js", ".ts"]):
                score += 1.0

            r["hybrid_score"] = score

        return results

    # -------------------------
    # Generic Keyword Extraction
    # -------------------------
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Repo-agnostic keyword extraction.
        No domain hardcoding.
        """
        words = re.findall(r"\b[a-zA-Z]{4,}\b", query.lower())

        stopwords = {
            "explain", "about", "this", "that", "repo",
            "implementation", "system", "code", "pattern"
        }

        keywords = [w for w in words if w not in stopwords]
        return keywords