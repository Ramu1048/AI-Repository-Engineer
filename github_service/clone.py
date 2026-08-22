"""
Git clone logic for shallow-cloning repositories.

Clones into data/repositories/{repository_id}/ using --depth 1.
Skips cloning if the directory already exists with the same repository_id.
"""

import logging
import os
import shutil
import subprocess

from exceptions import CloneError, CloneTimeoutError

logger = logging.getLogger(__name__)

# Default timeout for git clone (in seconds)
DEFAULT_CLONE_TIMEOUT = 120


def clone_repository(
    owner: str,
    repository: str,
    repository_id: str,
    default_branch: str = "main",
    base_dir: str = "data/repositories",
    timeout: int = DEFAULT_CLONE_TIMEOUT,
    token: str | None = None,
) -> str:
    """
    Shallow-clone a GitHub repository to a local directory.

    Args:
        owner: Repository owner.
        repository: Repository name.
        repository_id: Deterministic ID (owner_repo_sha) used as the directory name.
        default_branch: Branch to clone.
        base_dir: Parent directory for all cloned repos.
        timeout: Maximum seconds to wait for clone to complete.
        token: Optional GitHub token for private repos.

    Returns:
        Absolute path to the cloned repository directory.

    Raises:
        CloneError: Git clone failed (network, auth, etc.).
        CloneTimeoutError: Clone exceeded the configured timeout.
    """
    clone_dir = os.path.abspath(os.path.join(base_dir, repository_id))

    # Skip if already cloned
    if os.path.isdir(clone_dir):
        git_dir = os.path.join(clone_dir, ".git")
        if os.path.isdir(git_dir):
            logger.info(f"Repository already cloned at {clone_dir}, skipping.")
            return clone_dir
        else:
            # Directory exists but isn't a git repo — remove and re-clone
            logger.warning(f"Directory {clone_dir} exists but is not a git repo. Removing.")
            shutil.rmtree(clone_dir)

    # Ensure parent directory exists
    os.makedirs(base_dir, exist_ok=True)

    # Build clone URL (with optional token for private repos)
    if token:
        clone_url = f"https://{token}@github.com/{owner}/{repository}.git"
    else:
        clone_url = f"https://github.com/{owner}/{repository}.git"

    cmd = [
        "git", "clone",
        "--depth", "1",
        "--branch", default_branch,
        clone_url,
        clone_dir,
    ]

    logger.info(f"Cloning {owner}/{repository} (branch: {default_branch}) into {clone_dir}...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            # Clean up any partial clone
            if os.path.exists(clone_dir):
                shutil.rmtree(clone_dir, ignore_errors=True)

            stderr = result.stderr.strip()
            logger.error(f"Clone failed: {stderr}")
            raise CloneError(owner, repository, stderr)

        logger.info(f"Successfully cloned to {clone_dir}")
        return clone_dir

    except subprocess.TimeoutExpired:
        # Clean up partial clone
        if os.path.exists(clone_dir):
            shutil.rmtree(clone_dir, ignore_errors=True)
        logger.error(f"Clone timed out after {timeout}s")
        raise CloneTimeoutError(owner, repository, timeout)

    except OSError as e:
        if os.path.exists(clone_dir):
            shutil.rmtree(clone_dir, ignore_errors=True)
        raise CloneError(owner, repository, f"OS error: {e}")
