"""
Document Builder

Converts raw files into structured RAG documents with metadata.
Metadata is CRUCIAL for advanced retrieval filtering.
"""

from pathlib import Path
from typing import List, Dict
from .language_detector import LanguageDetector


class DocumentBuilder:
    def __init__(self):
        self.language_detector = LanguageDetector()

    def build_documents(self, files: List[Path], repo_name: str) -> List[Dict]:
        """
        Convert files into structured documents.
        Each document contains:
        - content
        - metadata (file_path, language, repo, etc.)
        """
        documents = []

        for file_path in files:
            try:
                content = file_path.read_text(
                    encoding="utf-8",
                    errors="ignore"  # Important for large repos
                )

                language = self.language_detector.detect_language(file_path)

                doc = {
                    "content": content,
                    "metadata": {
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "language": language,
                        "repo": repo_name,
                        "extension": file_path.suffix
                    }
                }

                documents.append(doc)

            except Exception as e:
                print(f"[WARN] Skipping file {file_path}: {e}")
                continue

        print(f"[INFO] Documents created: {len(documents)}")
        return documents