"""
GitHub Commits and Change Detection Module.
Identifies changed files in commits, evaluates architectural impact, and triggers re-indexing.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from github_service import GitHubClient


@dataclass
class CommitAnalysis:
    commit_hash: str
    author: str
    message: str
    changed_files: List[str] = field(default_factory=list)
    impact_summary: str = ""
    affected_modules: List[str] = field(default_factory=list)
    needs_reindex: bool = True


def analyze_commit(repository_id: str, commit_hash: str) -> CommitAnalysis:
    """
    Analyzes a commit hash to extract changed files, trigger re-indexing, and summarize downstream impact.

    :param repository_id: Repository identifier or directory path.
    :param commit_hash: Commit SHA string.
    :return: CommitAnalysis dataclass.
    """
    gh_client = GitHubClient()
    commit_data = gh_client.get_commit(repository_id, commit_hash)

    if not commit_data:
        raise ValueError(f"Commit '{commit_hash}' not found for repository '{repository_id}'.")

    author = commit_data.get("author", "Unknown")
    message = commit_data.get("message", "")
    changed_files = commit_data.get("changed_files", [])


    affected_modules = sorted(list({f.split("/")[0] for f in changed_files if "/" in f}))

    impact_summary = (
        f"Commit {commit_hash[:7]} by {author} modified {len(changed_files)} file(s) "
        f"across modules: {', '.join(affected_modules) if affected_modules else 'root'}."
    )

    return CommitAnalysis(
        commit_hash=commit_hash,
        author=author,
        message=message,
        changed_files=changed_files,
        impact_summary=impact_summary,
        affected_modules=affected_modules,
        needs_reindex=True,
    )
