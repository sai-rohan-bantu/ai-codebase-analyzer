"""
Production-Grade Multi-Language AST Parser
Supports: Java, Python, JavaScript, TypeScript, HTML
Zero-cost (Tree-sitter based)
"""

from tree_sitter_languages import get_parser


class ASTParser:
    def __init__(self):
        """
        Initialize and cache parsers for performance.
        Creating parser repeatedly is expensive.
        """
        self.parsers = {
            "python": self._safe_get_parser("python"),
            "java": self._safe_get_parser("java"),
            "javascript": self._safe_get_parser("javascript"),
            "typescript": self._safe_get_parser("typescript"),
            "react": self._safe_get_parser("tsx"),
            "html": self._safe_get_parser("html"),
        }

    def _safe_get_parser(self, language: str):
        """Safely load parser to avoid runtime crashes"""
        try:
            return get_parser(language)
        except Exception:
            return None

    def get_tree(self, content: str, language: str):
        """
        Returns AST tree for supported languages.
        Returns None for unsupported languages (markdown, css, etc.)
        """
        parser = self.parsers.get(language)

        if not parser or not content.strip():
            return None

        try:
            tree = parser.parse(bytes(content, "utf-8"))
            return tree
        except Exception:
            return None