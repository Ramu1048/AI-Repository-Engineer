"""Tests for github_service/clone.py — git clone logic and error handling."""

import os
import pytest
from unittest.mock import patch, MagicMock
import subprocess
import tempfile
import shutil

from github_service.clone import clone_repository
from exceptions import CloneError, CloneTimeoutError


@pytest.fixture
def temp_base_dir(tmp_path):
    """Provide a temporary base directory for cloning."""
    return str(tmp_path / "repositories")


class TestCloneRepository:
    """Test suite for clone_repository()."""

    @patch("github_service.clone.subprocess.run")
    def test_successful_clone(self, mock_run, temp_base_dir):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = clone_repository(
            owner="octocat",
            repository="Hello-World",
            repository_id="octocat_Hello-World_abc1234",
            default_branch="master",
            base_dir=temp_base_dir,
        )

        assert result.endswith("octocat_Hello-World_abc1234")
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "git" in cmd
        assert "--depth" in cmd
        assert "1" in cmd

    @patch("github_service.clone.subprocess.run")
    def test_clone_failure_raises_clone_error(self, mock_run, temp_base_dir):
        mock_run.return_value = MagicMock(
            returncode=128,
            stdout="",
            stderr="fatal: repository not found",
        )

        with pytest.raises(CloneError, match="repository not found"):
            clone_repository(
                owner="nonexistent",
                repository="repo",
                repository_id="nonexistent_repo_abc1234",
                base_dir=temp_base_dir,
            )

    @patch("github_service.clone.subprocess.run")
    def test_clone_timeout_raises_timeout_error(self, mock_run, temp_base_dir):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="git clone", timeout=120)

        with pytest.raises(CloneTimeoutError) as exc_info:
            clone_repository(
                owner="huge",
                repository="monorepo",
                repository_id="huge_monorepo_abc1234",
                base_dir=temp_base_dir,
                timeout=120,
            )
        assert exc_info.value.timeout_seconds == 120

    def test_skip_if_already_cloned(self, temp_base_dir):
        """If directory exists with .git folder, skip cloning."""
        repo_dir = os.path.join(temp_base_dir, "owner_repo_abc1234")
        git_dir = os.path.join(repo_dir, ".git")
        os.makedirs(git_dir, exist_ok=True)

        # Should return without calling git
        with patch("github_service.clone.subprocess.run") as mock_run:
            result = clone_repository(
                owner="owner",
                repository="repo",
                repository_id="owner_repo_abc1234",
                base_dir=temp_base_dir,
            )
            mock_run.assert_not_called()
            assert result == os.path.abspath(repo_dir)

    @patch("github_service.clone.subprocess.run")
    def test_reclone_if_dir_exists_without_git(self, mock_run, temp_base_dir):
        """If directory exists but has no .git, remove and re-clone."""
        repo_dir = os.path.join(temp_base_dir, "owner_repo_abc1234")
        os.makedirs(repo_dir, exist_ok=True)

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = clone_repository(
            owner="owner",
            repository="repo",
            repository_id="owner_repo_abc1234",
            base_dir=temp_base_dir,
        )

        mock_run.assert_called_once()

    @patch("github_service.clone.subprocess.run")
    def test_cleanup_on_failure(self, mock_run, temp_base_dir):
        """Partial clone directory should be cleaned up on failure."""
        repo_dir = os.path.join(temp_base_dir, "owner_repo_abc1234")

        def side_effect(*args, **kwargs):
            # Simulate git creating the directory then failing
            os.makedirs(repo_dir, exist_ok=True)
            return MagicMock(returncode=128, stdout="", stderr="fatal: error")

        mock_run.side_effect = side_effect

        with pytest.raises(CloneError):
            clone_repository(
                owner="owner",
                repository="repo",
                repository_id="owner_repo_abc1234",
                base_dir=temp_base_dir,
            )

        # Directory should be cleaned up
        assert not os.path.exists(repo_dir)
