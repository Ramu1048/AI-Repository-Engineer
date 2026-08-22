"""Tests for ingestion/loader.py — file loading, language detection, and skip logic."""

import os
import pytest

from ingestion.loader import (
    LoadedFile,
    load_repository_files,
    detect_language,
    detect_primary_languages,
)
from exceptions import EmptyRepositoryError


class TestDetectLanguage:
    """Test language detection from file extensions."""

    @pytest.mark.parametrize("path,expected", [
        ("main.py", "python"),
        ("app.js", "javascript"),
        ("index.ts", "typescript"),
        ("Main.java", "java"),
        ("main.go", "go"),
        ("lib.rs", "rust"),
        ("src/widget.cpp", "cpp"),
        ("include/header.h", "c"),
        ("include/header.hpp", "cpp"),
        ("src/core.c", "c"),
        ("README.md", "markdown"),
        ("notes.txt", "text"),
        ("config.yaml", "yaml"),
        ("config.yml", "yaml"),
        ("data.json", "json"),
        ("pom.xml", "xml"),
        ("Cargo.toml", "toml"),
        ("robot.xacro", "xml"),
        ("robot.urdf", "xml"),
        ("nav.launch.py", "python"),
    ])
    def test_language_mapping(self, path, expected):
        assert detect_language(path) == expected

    def test_unknown_extension(self):
        assert detect_language("file.xyz") == "unknown"


class TestDetectPrimaryLanguages:
    """Test primary language detection by file distribution."""

    def test_single_language(self):
        files = [
            LoadedFile("a.py", "", "python", 100),
            LoadedFile("b.py", "", "python", 100),
            LoadedFile("c.py", "", "python", 100),
        ]
        result = detect_primary_languages(files)
        assert result == ["python"]

    def test_multiple_languages_ordered(self):
        files = [
            LoadedFile("a.py", "", "python", 100),
            LoadedFile("b.py", "", "python", 100),
            LoadedFile("c.js", "", "javascript", 100),
            LoadedFile("d.go", "", "go", 100),
        ]
        result = detect_primary_languages(files)
        assert result[0] == "python"
        assert "javascript" in result
        assert "go" in result

    def test_non_code_excluded(self):
        """Markdown, JSON, etc. should not appear as primary languages."""
        files = [
            LoadedFile("a.py", "", "python", 100),
            LoadedFile("README.md", "", "markdown", 100),
            LoadedFile("config.json", "", "json", 100),
        ]
        result = detect_primary_languages(files)
        assert result == ["python"]

    def test_only_non_code_falls_back(self):
        """If only non-code files exist, fall back to listing all."""
        files = [
            LoadedFile("README.md", "", "markdown", 100),
            LoadedFile("config.json", "", "json", 100),
        ]
        result = detect_primary_languages(files)
        assert len(result) > 0


class TestLoadRepositoryFiles:
    """Test the full file loading pipeline with a temp directory structure."""

    @pytest.fixture
    def mock_repo(self, tmp_path):
        """Create a temporary repository with various file types."""
        repo_id = "test_owner_test_repo_abc1234"
        repo_dir = tmp_path / repo_id

        # Create Python files
        src_dir = repo_dir / "src"
        src_dir.mkdir(parents=True)
        (src_dir / "main.py").write_text("print('hello')", encoding="utf-8")
        (src_dir / "utils.py").write_text("def helper(): pass", encoding="utf-8")

        # Create a markdown file
        (repo_dir / "README.md").write_text("# Test Repo", encoding="utf-8")

        # Create a JSON config
        (repo_dir / "config.json").write_text('{"key": "value"}', encoding="utf-8")

        # Create a file that should be excluded (image)
        assets_dir = repo_dir / "assets"
        assets_dir.mkdir()
        (assets_dir / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        # Create a file in excluded directory
        pycache_dir = repo_dir / "__pycache__"
        pycache_dir.mkdir()
        (pycache_dir / "main.cpython-311.pyc").write_bytes(b"\x00" * 50)

        # Create a .git directory (should be excluded)
        git_dir = repo_dir / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main", encoding="utf-8")

        # Create a binary file that looks like text extension but can't decode
        (repo_dir / "binary.py").write_bytes(b"\xff\xfe" + b"\x00" * 100)

        return repo_id, str(tmp_path)

    def test_loads_expected_files(self, mock_repo):
        repo_id, base_dir = mock_repo
        files = load_repository_files(repo_id, base_dir)

        paths = {f.file_path for f in files}
        assert "src/main.py" in paths
        assert "src/utils.py" in paths
        assert "README.md" in paths
        assert "config.json" in paths

    def test_excludes_image_files(self, mock_repo):
        repo_id, base_dir = mock_repo
        files = load_repository_files(repo_id, base_dir)

        paths = {f.file_path for f in files}
        assert "assets/logo.png" not in paths

    def test_excludes_pycache(self, mock_repo):
        repo_id, base_dir = mock_repo
        files = load_repository_files(repo_id, base_dir)

        paths = {f.file_path for f in files}
        for p in paths:
            assert "__pycache__" not in p

    def test_excludes_git_dir(self, mock_repo):
        repo_id, base_dir = mock_repo
        files = load_repository_files(repo_id, base_dir)

        paths = {f.file_path for f in files}
        for p in paths:
            assert ".git" not in p.split("/")

    def test_file_content_is_loaded(self, mock_repo):
        repo_id, base_dir = mock_repo
        files = load_repository_files(repo_id, base_dir)

        main_file = [f for f in files if f.file_path == "src/main.py"][0]
        assert main_file.content == "print('hello')"
        assert main_file.language == "python"
        assert main_file.size_bytes > 0

    def test_handles_binary_decode_gracefully(self, mock_repo):
        """Binary files that pass extension check but fail UTF-8 decode should be skipped."""
        repo_id, base_dir = mock_repo
        files = load_repository_files(repo_id, base_dir)

        paths = {f.file_path for f in files}
        # binary.py should be skipped due to decode error
        assert "binary.py" not in paths

    def test_missing_directory_raises_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_repository_files("nonexistent_repo_abc1234", str(tmp_path))

    def test_empty_repo_raises_error(self, tmp_path):
        """A repo with no indexable files should raise EmptyRepositoryError."""
        repo_id = "empty_owner_empty_repo_abc1234"
        repo_dir = tmp_path / repo_id
        repo_dir.mkdir()
        # Only create non-indexable files
        (repo_dir / "logo.png").write_bytes(b"\x89PNG" + b"\x00" * 50)

        with pytest.raises(EmptyRepositoryError):
            load_repository_files(repo_id, str(tmp_path))

    def test_file_count(self, mock_repo):
        repo_id, base_dir = mock_repo
        files = load_repository_files(repo_id, base_dir)

        # Should have at least main.py, utils.py, README.md, config.json
        assert len(files) >= 4


class TestLoadedFileSize:
    """Test that oversized files are properly skipped."""

    def test_oversized_file_skipped(self, tmp_path):
        repo_id = "test_owner_test_repo_abc1234"
        repo_dir = tmp_path / repo_id
        repo_dir.mkdir(parents=True)

        # Create a small file that should be indexed
        (repo_dir / "small.py").write_text("x = 1", encoding="utf-8")

        # Create a file that exceeds the 2MB default limit
        big_content = "x" * (3 * 1024 * 1024)  # 3MB
        (repo_dir / "huge.py").write_text(big_content, encoding="utf-8")

        files = load_repository_files(repo_id, str(tmp_path))

        paths = {f.file_path for f in files}
        assert "small.py" in paths
        assert "huge.py" not in paths
