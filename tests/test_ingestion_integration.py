r"""
End-to-end integration test — full pipeline against a real GitHub repo.

Uses octocat/Hello-World (tiny, stable, public) to verify the complete flow:
URL → metadata → clone → filter → load → RepositoryMetadata + list[LoadedFile]

Run with:  .venv\Scripts\pytest tests\test_integration.py -v -m integration
"""

import os
import shutil
import pytest

from github_service.models import parse_github_url, generate_repository_id
from github_service.client import GitHubClient
from github_service.clone import clone_repository
from ingestion.loader import load_repository_files, detect_primary_languages
from ingestion.metadata import build_metadata, RepositoryMetadata


@pytest.mark.integration
class TestFullPipeline:
    """End-to-end test of the complete ingestion pipeline."""

    TEST_URL = "https://github.com/octocat/Hello-World"
    BASE_DIR = "data/repositories"

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up any cloned repos after each test."""
        yield
        # No auto-cleanup — leave it for debugging. The test is idempotent.

    def test_full_ingestion_pipeline(self):
        """
        Complete pipeline test:
        1. Parse URL
        2. Fetch metadata from GitHub API
        3. Clone repo
        4. Load and filter files
        5. Build RepositoryMetadata
        """

        # Step 1: Parse URL
        repo_url = parse_github_url(self.TEST_URL)
        assert repo_url.owner == "octocat"
        assert repo_url.repository == "Hello-World"

        # Step 2: Fetch metadata from GitHub API
        client = GitHubClient()
        github_meta = client.fetch_repo_metadata(repo_url.owner, repo_url.repository)

        assert github_meta["latest_commit_sha"]  # non-empty
        assert github_meta["default_branch"]  # non-empty
        assert isinstance(github_meta["stargazers_count"], int)

        # Step 3: Generate repository_id and clone
        commit_sha = github_meta["latest_commit_sha"]
        repository_id = generate_repository_id(
            repo_url.owner, repo_url.repository, commit_sha
        )
        assert repository_id.startswith("octocat_Hello-World_")
        assert len(repository_id.split("_")) >= 3

        clone_path = clone_repository(
            owner=repo_url.owner,
            repository=repo_url.repository,
            repository_id=repository_id,
            default_branch=github_meta["default_branch"],
            base_dir=self.BASE_DIR,
        )

        assert os.path.isdir(clone_path)
        assert os.path.isdir(os.path.join(clone_path, ".git"))

        # Step 4: Load and filter files
        loaded_files = load_repository_files(repository_id, self.BASE_DIR)

        assert len(loaded_files) > 0
        for lf in loaded_files:
            assert isinstance(lf.file_path, str)
            assert isinstance(lf.content, str)
            assert isinstance(lf.language, str)
            assert isinstance(lf.size_bytes, int)
            assert lf.size_bytes > 0

        # Step 5: Build metadata
        # Count total files for metadata (re-walk to get total including skipped)
        total_files = sum(
            len(files)
            for _, _, files in os.walk(clone_path)
        )
        skipped_files = total_files - len(loaded_files)

        metadata = build_metadata(
            repository_id=repository_id,
            owner=repo_url.owner,
            repository=repo_url.repository,
            github_meta=github_meta,
            loaded_files=loaded_files,
            file_count_total=total_files,
            file_count_skipped=skipped_files,
        )

        assert isinstance(metadata, RepositoryMetadata)
        assert metadata.repository_id == repository_id
        assert metadata.owner == "octocat"
        assert metadata.repository == "Hello-World"
        assert metadata.default_branch == github_meta["default_branch"]
        assert metadata.commit_sha == commit_sha
        assert metadata.file_count_indexed == len(loaded_files)
        assert metadata.file_count_total == total_files
        assert metadata.file_count_skipped == skipped_files
        assert isinstance(metadata.primary_languages, list)

        # Verify to_dict works
        meta_dict = metadata.to_dict()
        assert isinstance(meta_dict, dict)
        assert meta_dict["repository_id"] == repository_id

        # Print summary for manual inspection
        print("\n" + "=" * 60)
        print("INTEGRATION TEST — Full Pipeline Result")
        print("=" * 60)
        print(f"Repository ID:     {metadata.repository_id}")
        print(f"Owner:             {metadata.owner}")
        print(f"Repository:        {metadata.repository}")
        print(f"Description:       {metadata.description}")
        print(f"Primary Languages: {metadata.primary_languages}")
        print(f"Default Branch:    {metadata.default_branch}")
        print(f"Commit SHA:        {metadata.commit_sha}")
        print(f"Stars:             {metadata.stargazers_count}")
        print(f"Files Total:       {metadata.file_count_total}")
        print(f"Files Indexed:     {metadata.file_count_indexed}")
        print(f"Files Skipped:     {metadata.file_count_skipped}")
        print("-" * 60)
        print("Loaded files:")
        for lf in loaded_files:
            print(f"  {lf.file_path:40s} [{lf.language:12s}] {lf.size_bytes:>6d} bytes")
        print("=" * 60)
