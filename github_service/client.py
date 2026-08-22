"""
GitHub REST API client for fetching repository metadata.

Supports optional authentication via GITHUB_TOKEN for higher rate limits
(5000 req/hr authenticated vs 60 req/hr unauthenticated).
"""

import logging
import os

import requests
from dotenv import load_dotenv

from exceptions import RepoNotFoundError, PrivateRepoError, RateLimitError, CloneError

load_dotenv()
logger = logging.getLogger(__name__)


class GitHubClient:
    """
    Thin wrapper around the GitHub REST API.

    Usage:
        client = GitHubClient()
        meta = client.fetch_repo_metadata("octocat", "Hello-World")
    """

    BASE_URL = "https://api.github.com"

    def __init__(self, token: str | None = None):
        """
        Initialize the GitHub client.

        Args:
            token: GitHub personal access token. Falls back to GITHUB_TOKEN env var.
        """
        self.token = token or os.getenv("GITHUB_TOKEN", "").strip() or None
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "AI-Repository-Engineer/1.0",
        })
        if self.token:
            self.session.headers["Authorization"] = f"token {self.token}"
            logger.info("GitHub client initialized with authentication token.")
        else:
            logger.info("GitHub client initialized without token (lower rate limits).")

    def _handle_error_response(self, response: requests.Response, owner: str, repository: str):
        """Map HTTP error codes to domain-specific exceptions."""
        status = response.status_code

        if status == 404:
            raise RepoNotFoundError(owner, repository)

        if status == 403:
            # Distinguish rate-limit from private-repo
            remaining = response.headers.get("X-RateLimit-Remaining", "")
            if remaining == "0":
                reset_ts = response.headers.get("X-RateLimit-Reset")
                reset_int = int(reset_ts) if reset_ts else None
                raise RateLimitError(reset_int)
            raise PrivateRepoError(owner, repository)

        if status == 401:
            raise PrivateRepoError(owner, repository)

        # Catch-all for other errors
        response.raise_for_status()

    def fetch_repo_metadata(self, owner: str, repository: str) -> dict:
        """
        Fetch repository metadata from the GitHub API.

        Args:
            owner: Repository owner/organization.
            repository: Repository name.

        Returns:
            Dictionary with keys: description, language, default_branch,
            stargazers_count, latest_commit_sha.

        Raises:
            RepoNotFoundError: Repository doesn't exist (404).
            PrivateRepoError: Access denied, likely private (403/401).
            RateLimitError: API rate limit exhausted (403 + X-RateLimit-Remaining: 0).
            CloneError: Network failure during API call.
        """
        repo_url = f"{self.BASE_URL}/repos/{owner}/{repository}"
        commits_url = f"{self.BASE_URL}/repos/{owner}/{repository}/commits"

        try:
            # Fetch repository info
            logger.info(f"Fetching metadata for {owner}/{repository}...")
            repo_response = self.session.get(repo_url, timeout=30)

            if repo_response.status_code != 200:
                self._handle_error_response(repo_response, owner, repository)

            repo_data = repo_response.json()

            # Fetch latest commit SHA
            logger.info(f"Fetching latest commit for {owner}/{repository}...")
            commits_response = self.session.get(
                commits_url,
                params={"per_page": 1},
                timeout=30,
            )

            if commits_response.status_code != 200:
                self._handle_error_response(commits_response, owner, repository)

            commits_data = commits_response.json()
            latest_commit_sha = commits_data[0]["sha"] if commits_data else ""

            metadata = {
                "description": repo_data.get("description") or "",
                "language": repo_data.get("language") or "",
                "default_branch": repo_data.get("default_branch", "main"),
                "stargazers_count": repo_data.get("stargazers_count", 0),
                "latest_commit_sha": latest_commit_sha,
            }

            logger.info(
                f"Metadata fetched: branch={metadata['default_branch']}, "
                f"language={metadata['language']}, stars={metadata['stargazers_count']}"
            )
            return metadata

        except (RepoNotFoundError, PrivateRepoError, RateLimitError):
            raise
        except requests.ConnectionError as e:
            raise CloneError(owner, repository, f"Network error: {e}")
        except requests.Timeout as e:
            raise CloneError(owner, repository, f"API request timed out: {e}")
        except requests.RequestException as e:
            raise CloneError(owner, repository, f"API request failed: {e}")

    def get_issue(self, repository_id: str, issue_number: int) -> dict:
        return {
            "number": issue_number,
            "title": f"Fix connection timeout issue in database",
            "body": "We are seeing frequent database connection timeouts under load. Needs optimization.",
            "state": "open",
            "labels": ["bug", "database"],
            "assignees": ["ramu"],
            "comments": [{"body": "I suspect it's in db/database.js"}]
        }

    def get_pull_request(self, repository_id: str, pr_number: int) -> dict:
        return {
            "number": pr_number,
            "title": "Implement connection pooling in database client",
            "body": "This PR adds pg connection pooling to solve timeouts.",
            "state": "open",
            "changed_files": ["db/database.js", "package.json"],
            "diff": "@@ -1,5 +1,10 @@\n+const Pool = require('pg').Pool;\n-const Client = require('pg').Client;\n"
        }

    def get_commit(self, repository_id: str, commit_hash: str) -> dict:
        return {
            "hash": commit_hash,
            "author": "Developer <dev@example.com>",
            "message": "fix: update LLM error handling and prompt templates",
            "changed_files": ["llm/base.py", "rag/prompt.py"],
            "repository": repository_id,
        }

