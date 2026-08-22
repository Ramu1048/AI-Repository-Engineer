"""
Repository metadata builder — combines GitHub API data with local clone stats.
"""

from dataclasses import dataclass, field, asdict
from typing import Any

from ingestion.loader import LoadedFile, detect_primary_languages


@dataclass
class RepositoryMetadata:
    """
    Complete metadata for an ingested repository.

    Combines GitHub API metadata with local file statistics.
    This is the top-level output of the ingestion pipeline.
    """
    repository_id: str
    owner: str
    repository: str
    description: str
    primary_languages: list[str]
    default_branch: str
    commit_sha: str
    stargazers_count: int
    file_count_total: int
    file_count_indexed: int
    file_count_skipped: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary."""
        return asdict(self)


def build_metadata(
    repository_id: str,
    owner: str,
    repository: str,
    github_meta: dict,
    loaded_files: list[LoadedFile],
    file_count_total: int,
    file_count_skipped: int,
) -> RepositoryMetadata:
    """
    Build a RepositoryMetadata object from API data and local stats.

    Args:
        repository_id: Deterministic repo ID (owner_repo_sha).
        owner: Repository owner.
        repository: Repository name.
        github_meta: Dict from GitHubClient.fetch_repo_metadata().
        loaded_files: List of LoadedFile objects from the loader.
        file_count_total: Total files found in the clone.
        file_count_skipped: Files that were skipped (filtered out or decode errors).

    Returns:
        A fully populated RepositoryMetadata object.
    """
    primary_languages = detect_primary_languages(loaded_files)

    return RepositoryMetadata(
        repository_id=repository_id,
        owner=owner,
        repository=repository,
        description=github_meta.get("description", ""),
        primary_languages=primary_languages,
        default_branch=github_meta.get("default_branch", "main"),
        commit_sha=github_meta.get("latest_commit_sha", ""),
        stargazers_count=github_meta.get("stargazers_count", 0),
        file_count_total=file_count_total,
        file_count_indexed=len(loaded_files),
        file_count_skipped=file_count_skipped,
    )
