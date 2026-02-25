"""
Production-Grade Query-Aware Context Aggregator (Copilot-Style)

Features:
- Repo-agnostic (no hardcoding for specific repos)
- Smart module extraction (avoids repo root grouping)
- Query-aware module scoring
- Global vs Local query routing (architecture vs specific feature)
- Explanation-aware chunk ranking (class/interface > main/demo files)
- Works for:
  - Multi-module repos
  - Monorepos
  - Random GitHub repos
  - Local codebases
"""

from collections import defaultdict
from typing import List, Dict
import re


class ContextAggregator:
    def __init__(self):
        # Generic folders to ignore as logical modules
        self.ignore_folders = {
            "src", "main", "java", "python", "js", "ts",
            "com", "org", "example", "repos", "test", "tests"
        }

        # Boilerplate/demo files (repo-agnostic)
        self.boilerplate_files = {
            "main.java", "app.java", "index.js", "main.py",
            "app.py", "index.ts", "main.ts"
        }

    # =====================================================
    # PUBLIC METHOD (MAIN ENTRY)
    # =====================================================
    def aggregate(
        self,
        query: str,
        chunks: List[Dict],
        final_top_k: int = 5
    ) -> List[Dict]:
        """
        Full aggregation pipeline:
        1. Extract query keywords
        2. Group chunks by logical module
        3. Detect global vs local query
        4. Score modules (query-aware)
        5. Select best module or multi-module context
        6. Rank chunks for explanation quality
        """
        if not chunks:
            print("[AGGREGATOR] No chunks received.")
            return []

        query_keywords = self._extract_keywords(query)

        # Step 1: Group chunks by module
        module_groups = defaultdict(list)
        for chunk in chunks:
            file_path = chunk["metadata"].get("file_path", "")
            module = self._extract_module(file_path)
            module_groups[module].append(chunk)

        if not module_groups:
            return chunks[:final_top_k]

        # Step 2: Global vs Local query detection
        if self._is_global_query(query):
            print("[AGGREGATOR] Global query detected → using multi-module context")
            all_chunks = []
            for module_chunks in module_groups.values():
                all_chunks.extend(module_chunks)

            ranked = self._rank_for_explanation(all_chunks)
            return ranked[:final_top_k]

        # Step 3: Query-aware module scoring (local queries)
        module_scores = {}

        for module, module_chunks in module_groups.items():
            module_score = 0.0
            module_lower = module.lower()

            # 1️⃣ Query-Module alignment (strong signal, repo-agnostic)
            keyword_hits = sum(1 for kw in query_keywords if kw in module_lower)
            module_score += keyword_hits * 12.0

            # 2️⃣ Aggregate semantic scores
            for c in module_chunks:
                rerank = c.get("rerank_score", 0.0)
                hybrid = c.get("hybrid_score", 0.0)

                # Clamp extreme negatives so they don’t dominate
                normalized_rerank = max(rerank, -5.0)

                module_score += (hybrid * 1.0) + (normalized_rerank * 0.5)

            # 3️⃣ Coherence boost (more relevant chunks in same module)
            module_score += len(module_chunks) * 2.5

            # 4️⃣ Penalize generic/root-like modules
            if self._is_generic_module(module):
                module_score -= 10.0

            module_scores[module] = module_score

        # Step 4: Select best module (for focused queries)
        best_module = max(module_scores, key=module_scores.get)
        print(f"[AGGREGATOR] Selected module: {best_module}")

        best_chunks = module_groups[best_module]

        # Step 5: Explanation-aware ranking inside module
        ranked = self._rank_for_explanation(best_chunks)

        return ranked[:final_top_k]

    # =====================================================
    # EXPLANATION-AWARE RANKING (CRITICAL FOR COPILOT STYLE)
    # =====================================================
    def _rank_for_explanation(self, chunks: List[Dict]) -> List[Dict]:
        """
        Rank chunks for explanation quality:
        interface > class > method > text
        and penalize boilerplate/demo files.
        """

        def explanation_score(chunk: Dict) -> float:
            meta = chunk["metadata"]
            file_name = meta.get("file_name", "").lower()
            chunk_type = meta.get("chunk_type", "text_chunk")

            score = 0.0

            # Prefer architecture-relevant structures
            if chunk_type == "interface_declaration":
                score += 8.0
            elif chunk_type == "class_declaration":
                score += 6.0
            elif chunk_type == "method_declaration":
                score += 3.0
            else:
                score += 1.0

            # Penalize boilerplate/demo entry files (repo-agnostic)
            if file_name in self.boilerplate_files:
                score -= 5.0

            # Add semantic signals
            hybrid = chunk.get("hybrid_score", 0.0)
            rerank = chunk.get("rerank_score", 0.0)
            normalized_rerank = max(rerank, -5.0)

            score += hybrid * 1.0
            score += normalized_rerank * 0.5

            # Prefer larger semantic chunks (better context)
            start = meta.get("start_line", 0) or 0
            end = meta.get("end_line", 0) or 0
            size = max(0, end - start)
            score += min(size / 20.0, 3.0)

            return score

        return sorted(chunks, key=explanation_score, reverse=True)

    # =====================================================
    # SMART MODULE EXTRACTION (AVOIDS REPO ROOT ISSUE)
    # =====================================================
    def _extract_module(self, file_path: str) -> str:
        """
        Extract true logical module instead of repo root.

        Example:
        repos/SystemDesign-LLD-Behavioural/Strategy-Design-Pattern/src/...
        → returns: Strategy-Design-Pattern
        """
        if not file_path:
            return "unknown_module"

        path = file_path.replace("\\", "/")
        parts = [p for p in path.split("/") if p]

        if not parts:
            return "unknown_module"

        # Remove leading "repos" if present
        if parts[0].lower() == "repos" and len(parts) > 1:
            parts = parts[1:]

        # parts[0] = repo_name
        if len(parts) >= 2:
            repo_name = parts[0].lower()

            # Find first meaningful folder after repo name
            for i in range(1, len(parts)):
                folder = parts[i].lower()

                if (
                    folder not in self.ignore_folders
                    and folder != repo_name
                    and len(folder) > 3
                ):
                    return parts[i]

        # Fallback: parent directory near file
        if len(parts) >= 2:
            return parts[-2]

        return "unknown_module"

    # =====================================================
    # QUERY KEYWORD EXTRACTION (REPO-AGNOSTIC)
    # =====================================================
    def _extract_keywords(self, query: str) -> List[str]:
        words = re.findall(r"\b[a-zA-Z]{4,}\b", query.lower())

        stopwords = {
            "explain", "where", "what", "this", "that",
            "repo", "code", "implementation", "project",
            "about", "with", "from", "pattern"
        }

        return [w for w in words if w not in stopwords]

    # =====================================================
    # GLOBAL QUERY DETECTOR (ARCHITECTURE / OVERVIEW)
    # =====================================================
    def _is_global_query(self, query: str) -> bool:
        query = query.lower()
        global_keywords = [
            "architecture", "overall", "system", "design",
            "structure", "workflow", "overview"
        ]
        return any(k in query for k in global_keywords)

    # =====================================================
    # GENERIC MODULE DETECTOR (ROOT PENALTY)
    # =====================================================
    def _is_generic_module(self, module: str) -> bool:
        module_lower = module.lower()
        generic_indicators = [
            "repo", "project", "codebase", "src", "main"
        ]
        return any(ind in module_lower for ind in generic_indicators)