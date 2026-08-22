"""Tests for github_service/models.py — URL parsing and repository ID generation."""

import pytest

from github_service.models import parse_github_url, generate_repository_id, GitHubRepoURL
from exceptions import InvalidGitHubURLError


class TestParseGitHubURL:
    """Test suite for parse_github_url()."""

    # --- Valid URLs ---

    def test_basic_url(self):
        result = parse_github_url("https://github.com/octocat/Hello-World")
        assert result.owner == "octocat"
        assert result.repository == "Hello-World"

    def test_url_with_git_suffix(self):
        result = parse_github_url("https://github.com/octocat/Hello-World.git")
        assert result.owner == "octocat"
        assert result.repository == "Hello-World"

    def test_url_with_trailing_slash(self):
        result = parse_github_url("https://github.com/octocat/Hello-World/")
        assert result.owner == "octocat"
        assert result.repository == "Hello-World"

    def test_url_with_git_and_trailing_slash(self):
        result = parse_github_url("https://github.com/octocat/Hello-World.git/")
        assert result.owner == "octocat"
        assert result.repository == "Hello-World"

    def test_url_with_whitespace(self):
        result = parse_github_url("  https://github.com/octocat/Hello-World  ")
        assert result.owner == "octocat"
        assert result.repository == "Hello-World"

    def test_url_with_dots_in_name(self):
        result = parse_github_url("https://github.com/owner/repo.name")
        assert result.owner == "owner"
        assert result.repository == "repo.name"

    def test_url_with_underscores(self):
        result = parse_github_url("https://github.com/my_org/my_repo")
        assert result.owner == "my_org"
        assert result.repository == "my_repo"

    def test_url_numeric_repo(self):
        result = parse_github_url("https://github.com/user123/repo456")
        assert result.owner == "user123"
        assert result.repository == "repo456"

    # --- Invalid URLs ---

    def test_empty_string(self):
        with pytest.raises(InvalidGitHubURLError):
            parse_github_url("")

    def test_none_value(self):
        with pytest.raises(InvalidGitHubURLError):
            parse_github_url(None)

    def test_http_not_https(self):
        with pytest.raises(InvalidGitHubURLError, match="use HTTPS"):
            parse_github_url("http://github.com/owner/repo")

    def test_ssh_url(self):
        with pytest.raises(InvalidGitHubURLError, match="SSH"):
            parse_github_url("git@github.com:owner/repo.git")

    def test_gitlab_url(self):
        with pytest.raises(InvalidGitHubURLError, match="only GitHub"):
            parse_github_url("https://gitlab.com/owner/repo")

    def test_missing_repo_name(self):
        with pytest.raises(InvalidGitHubURLError):
            parse_github_url("https://github.com/owner")

    def test_missing_owner_and_repo(self):
        with pytest.raises(InvalidGitHubURLError):
            parse_github_url("https://github.com/")

    def test_random_string(self):
        with pytest.raises(InvalidGitHubURLError):
            parse_github_url("not-a-url-at-all")

    def test_url_with_extra_path(self):
        # URLs with extra path segments (like /tree/main) should not match
        with pytest.raises(InvalidGitHubURLError):
            parse_github_url("https://github.com/owner/repo/tree/main")


class TestGitHubRepoURLProperties:
    """Test computed properties on GitHubRepoURL."""

    def test_clone_url(self):
        url = GitHubRepoURL(owner="octocat", repository="Hello-World")
        assert url.clone_url == "https://github.com/octocat/Hello-World.git"

    def test_api_url(self):
        url = GitHubRepoURL(owner="octocat", repository="Hello-World")
        assert url.api_url == "https://api.github.com/repos/octocat/Hello-World"

    def test_str(self):
        url = GitHubRepoURL(owner="octocat", repository="Hello-World")
        assert str(url) == "https://github.com/octocat/Hello-World"


class TestGenerateRepositoryId:
    """Test suite for generate_repository_id()."""

    def test_basic_id(self):
        result = generate_repository_id("octocat", "Hello-World", "a1b2c3d4e5f6")
        assert result == "octocat_Hello-World_a1b2c3d"

    def test_short_sha(self):
        result = generate_repository_id("owner", "repo", "abc1234")
        assert result == "owner_repo_abc1234"

    def test_full_40_char_sha(self):
        sha = "a" * 40
        result = generate_repository_id("o", "r", sha)
        assert result == "o_r_aaaaaaa"

    def test_deterministic(self):
        """Same inputs always produce the same ID."""
        id1 = generate_repository_id("octocat", "Hello-World", "abc1234")
        id2 = generate_repository_id("octocat", "Hello-World", "abc1234")
        assert id1 == id2

    def test_different_commits_different_ids(self):
        id1 = generate_repository_id("owner", "repo", "abc1234")
        id2 = generate_repository_id("owner", "repo", "xyz9876")
        assert id1 != id2
