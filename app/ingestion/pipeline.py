"""
Full GitHub Ingestion Pipeline

Flow:
GitHub URL → Clone/Update → Scan Files → Build Documents
"""

from .github_loader import GitHubRepoLoader
from .file_scanner import FileScanner
from .document_builder import DocumentBuilder
from app.parsing.code_chunker import CodeChunker


class IngestionPipeline:
    def __init__(self):
        self.repo_loader = GitHubRepoLoader()
        self.file_scanner = FileScanner()
        self.document_builder = DocumentBuilder()
        self.chunker = CodeChunker()

    def ingest_from_github(self, repo_url: str):
        """
        Main entry point for GitHub ingestion.
        Returns structured documents ready for:
        - Parsing
        - Chunking
        - Embeddings
        """
        print(f"[INFO] Starting ingestion for: {repo_url}")

        # Step 1: Clone or update repo
        repo_path = self.repo_loader.clone_or_update_repo(repo_url)
        repo_name = repo_path.name

        # Step 2: Scan relevant files
        files = self.file_scanner.scan_repository(repo_path)

        # Step 3: Convert to RAG documents
        documents = self.document_builder.build_documents(files, repo_name)

        # Step 4: Semantic AST Chunking
        semantic_chunks = self.chunker.chunk_documents(documents)

        return semantic_chunks