"""
Custom exceptions for the AI Repository Engineer ingestion pipeline.

Each exception maps to a specific failure mode so the FastAPI layer (Member 5)
can return precise, user-facing error messages instead of generic 500s.
"""


class IngestionError(Exception):
    """Base exception for all ingestion-related errors."""
    pass


class InvalidGitHubURLError(IngestionError):
    """Raised when a GitHub URL is malformed or doesn't match the expected pattern."""

    def __init__(self, url: str, reason: str = ""):
        self.url = url
        detail = f" ({reason})" if reason else ""
        super().__init__(
            f"Invalid GitHub URL: '{url}'{detail}. "
            f"Expected format: https://github.com/OWNER/REPOSITORY"
        )


class RepoNotFoundError(IngestionError):
    """Raised when the GitHub API returns 404 for a repository."""

    def __init__(self, owner: str, repository: str):
        self.owner = owner
        self.repository = repository
        super().__init__(
            f"Repository not found: '{owner}/{repository}'. "
            f"Check that the owner and repository name are correct and the repo is public."
        )


class PrivateRepoError(IngestionError):
    """Raised when access is denied to a private repository (403 without rate-limit exhaustion)."""

    def __init__(self, owner: str, repository: str):
        self.owner = owner
        self.repository = repository
        super().__init__(
            f"Access denied to '{owner}/{repository}'. "
            f"The repository may be private. Provide a GITHUB_TOKEN with appropriate permissions."
        )


class RateLimitError(IngestionError):
    """Raised when the GitHub API rate limit has been exceeded."""

    def __init__(self, reset_timestamp: int | None = None):
        self.reset_timestamp = reset_timestamp
        reset_info = ""
        if reset_timestamp:
            import datetime
            reset_time = datetime.datetime.fromtimestamp(reset_timestamp, tz=datetime.timezone.utc)
            reset_info = f" Rate limit resets at {reset_time.isoformat()}."
        super().__init__(
            f"GitHub API rate limit exceeded.{reset_info} "
            f"Set GITHUB_TOKEN in .env for higher limits (5000/hr vs 60/hr)."
        )


class CloneError(IngestionError):
    """Raised when git clone fails (network error, auth failure, etc.)."""

    def __init__(self, owner: str, repository: str, reason: str = ""):
        self.owner = owner
        self.repository = repository
        detail = f": {reason}" if reason else ""
        super().__init__(
            f"Failed to clone '{owner}/{repository}'{detail}"
        )


class CloneTimeoutError(CloneError):
    """Raised when git clone exceeds the configured timeout."""

    def __init__(self, owner: str, repository: str, timeout_seconds: int):
        self.timeout_seconds = timeout_seconds
        super().__init__(
            owner, repository,
            f"Clone timed out after {timeout_seconds}s. The repository may be too large."
        )


class EmptyRepositoryError(IngestionError):
    """Raised when a cloned repository contains zero indexable files after filtering."""

    def __init__(self, owner: str, repository: str):
        self.owner = owner
        self.repository = repository
        super().__init__(
            f"Repository '{owner}/{repository}' has no indexable files after filtering. "
            f"It may be empty or contain only unsupported file types."
        )


class FileDecodeError(IngestionError):
    """Raised when a file cannot be decoded as UTF-8 (likely binary that slipped through filtering)."""

    def __init__(self, file_path: str, reason: str = ""):
        self.file_path = file_path
        detail = f": {reason}" if reason else ""
        super().__init__(
            f"Failed to decode file as UTF-8: '{file_path}'{detail}"
        )
