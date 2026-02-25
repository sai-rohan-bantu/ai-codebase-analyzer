"""
GitHub Repository Loader

Responsibilities:
- Clone GitHub repositories
- Manage local workspace
- Prepare repos for analysis pipeline

Industry Pattern:
Source Connector Layer (used in enterprise RAG systems)
"""

import shutil
from pathlib import Path
from git import Repo


class GitHubRepoLoader:
    def __init__(self, base_dir: str = "repos"):
        """
        base_dir: Directory where repositories will be cloned
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_repo_name(self, repo_url: str) -> str:
        """Extract repository name from GitHub URL"""
        return repo_url.rstrip("/").split("/")[-1].replace(".git", "")

    def clone_or_update_repo(self, repo_url: str) -> Path:
        """
        Clone repo if not exists, else pull latest changes.
        Production-friendly approach (better than always deleting).
        """
        repo_name = self._get_repo_name(repo_url)
        repo_path = self.base_dir / repo_name

        if repo_path.exists():
            # Update existing repo (incremental ingestion strategy)
            try:
                repo = Repo(repo_path)
                origin = repo.remotes.origin
                origin.pull()
                print(f"[INFO] Updated existing repo: {repo_name}")
            except Exception as e:
                print(f"[WARN] Failed to pull repo, recloning: {e}")
                shutil.rmtree(repo_path)
                Repo.clone_from(repo_url, repo_path)
        else:
            # Fresh clone
            Repo.clone_from(repo_url, repo_path)
            print(f"[INFO] Cloned new repo: {repo_name}")

        return repo_path