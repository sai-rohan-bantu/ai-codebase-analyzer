"""
Language Detection Module

Used for:
- Language-aware parsing
- Smart chunking strategies
- Metadata enrichment
"""

from pathlib import Path

EXTENSION_TO_LANGUAGE = {
    ".java": "java",
    ".py": "python",
    ".js": "javascript",
    ".jsx": "react",
    ".ts": "typescript",
    ".tsx": "react",
    ".html": "html",
    ".css": "css",
    ".md": "markdown",
    ".json": "config",
    ".yaml": "config",
    ".yml": "config"
}


class LanguageDetector:
    def detect_language(self, file_path: Path) -> str:
        """
        Detect programming language based on file extension.
        """
        return EXTENSION_TO_LANGUAGE.get(
            file_path.suffix.lower(),
            "unknown"
        )