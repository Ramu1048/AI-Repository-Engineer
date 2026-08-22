"""Tests for github_service/client.py — API metadata fetching and error handling."""

import pytest
from unittest.mock import patch, MagicMock

from github_service.client import GitHubClient
from exceptions import RepoNotFoundError, PrivateRepoError, RateLimitError, CloneError


@pytest.fixture
def client():
    """Client without token for testing."""
    return GitHubClient(token=None)


class TestFetchRepoMetadata:
    """Test the fetch_repo_metadata method with mocked HTTP responses."""

    def _mock_response(self, status_code=200, json_data=None, headers=None):
        """Helper to create a mock requests.Response."""
        mock = MagicMock()
        mock.status_code = status_code
        mock.json.return_value = json_data or {}
        mock.headers = headers or {}
        return mock

    @patch.object(GitHubClient, "__init__", lambda self, **kwargs: setattr(self, "session", MagicMock()) or setattr(self, "token", None))
    def test_successful_fetch(self):
        client = GitHubClient()
        repo_data = {
            "description": "My first repository on GitHub!",
            "language": "Python",
            "default_branch": "master",
            "stargazers_count": 2500,
        }
        commits_data = [{"sha": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"}]

        client.session.get.side_effect = [
            self._mock_response(200, repo_data),
            self._mock_response(200, commits_data),
        ]

        result = client.fetch_repo_metadata("octocat", "Hello-World")

        assert result["description"] == "My first repository on GitHub!"
        assert result["language"] == "Python"
        assert result["default_branch"] == "master"
        assert result["stargazers_count"] == 2500
        assert result["latest_commit_sha"] == "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"

    @patch.object(GitHubClient, "__init__", lambda self, **kwargs: setattr(self, "session", MagicMock()) or setattr(self, "token", None))
    def test_repo_not_found_404(self):
        client = GitHubClient()
        client.session.get.return_value = self._mock_response(404)

        with pytest.raises(RepoNotFoundError) as exc_info:
            client.fetch_repo_metadata("nonexistent", "repo")
        assert "nonexistent/repo" in str(exc_info.value)

    @patch.object(GitHubClient, "__init__", lambda self, **kwargs: setattr(self, "session", MagicMock()) or setattr(self, "token", None))
    def test_private_repo_403(self):
        client = GitHubClient()
        client.session.get.return_value = self._mock_response(
            403, headers={"X-RateLimit-Remaining": "50"}
        )

        with pytest.raises(PrivateRepoError) as exc_info:
            client.fetch_repo_metadata("private", "repo")
        assert "private/repo" in str(exc_info.value)

    @patch.object(GitHubClient, "__init__", lambda self, **kwargs: setattr(self, "session", MagicMock()) or setattr(self, "token", None))
    def test_rate_limit_403(self):
        client = GitHubClient()
        client.session.get.return_value = self._mock_response(
            403,
            headers={
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": "1700000000",
            },
        )

        with pytest.raises(RateLimitError) as exc_info:
            client.fetch_repo_metadata("any", "repo")
        assert exc_info.value.reset_timestamp == 1700000000

    @patch.object(GitHubClient, "__init__", lambda self, **kwargs: setattr(self, "session", MagicMock()) or setattr(self, "token", None))
    def test_network_error(self):
        import requests as req
        client = GitHubClient()
        client.session.get.side_effect = req.ConnectionError("DNS resolution failed")

        with pytest.raises(CloneError, match="Network error"):
            client.fetch_repo_metadata("owner", "repo")

    @patch.object(GitHubClient, "__init__", lambda self, **kwargs: setattr(self, "session", MagicMock()) or setattr(self, "token", None))
    def test_missing_description_returns_empty(self):
        client = GitHubClient()
        repo_data = {
            "description": None,
            "language": None,
            "default_branch": "main",
            "stargazers_count": 0,
        }
        commits_data = [{"sha": "abcdef1234567890"}]

        client.session.get.side_effect = [
            self._mock_response(200, repo_data),
            self._mock_response(200, commits_data),
        ]

        result = client.fetch_repo_metadata("owner", "repo")
        assert result["description"] == ""
        assert result["language"] == ""
