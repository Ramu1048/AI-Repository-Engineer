"""
Repository URL parsing, validation, and identity generation.

Accepts GitHub URLs in the form:
    https://github.com/OWNER/REPOSITORY[.git][/]

Produces a deterministic repository_id: owner_repository_commitHash
"""

import re
from dataclasses import dataclass

from exceptions import InvalidGitHubURLError


# Pattern: https://github.com/OWNER/REPO with optional .git and/or trailing slash
_GITHUB_URL_PATTERN = re.compile(
    r"^https://github\.com/"
    r"(?P<owner>[A-Za-z0-9\-_.]+)/"
    r"(?P<repository>[A-Za-z0-9\-_.]+?)"
    r"(?:\.git)?/?\s*$"
)


@dataclass(frozen=True)
class GitHubRepoURL:
    """Parsed and validated GitHub repository URL."""
    owner: str
    repository: str

    @property
    def clone_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.repository}.git"

    @property
    def api_url(self) -> str:
        return f"https://api.github.com/repos/{self.owner}/{self.repository}"

    def __str__(self) -> str:
        return f"https://github.com/{self.owner}/{self.repository}"


def parse_github_url(url: str) -> GitHubRepoURL:
    """
    Parse and validate a GitHub repository URL.

    Args:
        url: A GitHub URL like https://github.com/owner/repo

    Returns:
        GitHubRepoURL with parsed owner and repository.

    Raises:
        InvalidGitHubURLError: If the URL doesn't match the expected pattern.
    """
    if not url or not isinstance(url, str):
        raise InvalidGitHubURLError(str(url), "URL must be a non-empty string")

    url = url.strip()

    # Quick sanity checks for better error messages
    if not url.startswith("https://"):
        if url.startswith("http://"):
            raise InvalidGitHubURLError(url, "use HTTPS, not HTTP")
        if url.startswith("git@"):
            raise InvalidGitHubURLError(url, "SSH URLs are not supported, use HTTPS")
        raise InvalidGitHubURLError(url, "URL must start with https://")

    if "github.com" not in url:
        raise InvalidGitHubURLError(url, "only GitHub URLs are supported")

    match = _GITHUB_URL_PATTERN.match(url)
    if not match:
        raise InvalidGitHubURLError(
            url,
            "could not parse owner and repository from URL"
        )

    owner = match.group("owner")
    repository = match.group("repository")

    # Reject edge cases
    if owner.startswith(".") or owner.startswith("-"):
        raise InvalidGitHubURLError(url, f"invalid owner name: '{owner}'")
    if repository.startswith(".") or repository.startswith("-"):
        raise InvalidGitHubURLError(url, f"invalid repository name: '{repository}'")

    return GitHubRepoURL(owner=owner, repository=repository)


def generate_repository_id(owner: str, repository: str, commit_sha: str) -> str:
    """
    Generate a deterministic repository ID.

    Format: owner_repository_shortsha (7-char short SHA).
    This ID is the isolation key the whole system uses to avoid mixing
    data between repositories.

    Args:
        owner: GitHub repository owner.
        repository: GitHub repository name.
        commit_sha: Full or short commit SHA of the cloned HEAD.

    Returns:
        A string like 'octocat_Hello-World_a1b2c3d'.
    """
    short_sha = commit_sha[:7]
    return f"{owner}_{repository}_{short_sha}"
