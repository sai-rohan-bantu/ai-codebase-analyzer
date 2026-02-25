"""
Repository File Scanner

Responsibilities:
- Traverse repository files
- Filter supported languages
- Ignore noise (node_modules, build, .git, etc.)

This is CRITICAL for codebase analyzers.
"""

from pathlib import Path
from typing import List


SUPPORTED_EXTENSIONS = {
    ".java", ".py", ".js", ".jsx", ".ts", ".tsx",
    ".html", ".css", ".md", ".json", ".yaml", ".yml"
}

IGNORE_DIRECTORIES = {
    ".git",
    "node_modules",
    "dist",
    "build",
    "target",
    "__pycache__",
    ".idea",
    ".vscode",
    ".next"
}

IGNORE_FILES = {
    ".gitignore",
    ".env",
    ".DS_Store"
}


class FileScanner:
    def scan_repository(self, repo_path: Path) -> List[Path]:
        """
        Recursively scan repository and return valid files only.
        """
        valid_files = []

        for file_path in repo_path.rglob("*"):
            # Skip directories
            if file_path.is_dir():
                continue

            # Ignore unwanted folders
            if any(ignored in file_path.parts for ignored in IGNORE_DIRECTORIES):
                continue

            if file_path.name in IGNORE_FILES:
                continue

            # Filter by supported extensions
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                valid_files.append(file_path)

        print(f"[INFO] Total relevant files found: {len(valid_files)}")
        return valid_files