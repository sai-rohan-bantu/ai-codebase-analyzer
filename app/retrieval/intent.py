"""
Query Intent Detection (Post-Index Stage)

Used to adapt retrieval strategy:
- search: locate files/classes
- explain: deep semantic explanation
- hybrid: both (default for code assistants)
"""

class QueryIntentDetector:
    def __init__(self):
        self.search_keywords = [
            "where", "find", "locate", "which file", "search", "implemented"
        ]

        self.explain_keywords = [
            "explain", "how", "why", "logic", "flow", "working", "architecture"
        ]

    def detect(self, query: str) -> str:
        q = query.lower()

        is_search = any(k in q for k in self.search_keywords)
        is_explain = any(k in q for k in self.explain_keywords)

        if is_search and is_explain:
            return "hybrid"
        if is_search:
            return "search"
        if is_explain:
            return "explain"

        # Default for code assistants
        return "hybrid"