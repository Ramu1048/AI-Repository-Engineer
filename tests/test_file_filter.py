"""Tests for ingestion/file_filter.py — include/exclude rules and size limits."""

import pytest
from unittest.mock import patch

from ingestion.file_filter import FileFilter, INCLUDED_EXTENSIONS, EXCLUDED_DIRS


class TestFileFilterInclusion:
    """Test file inclusion rules."""

    @pytest.fixture
    def ff(self):
        return FileFilter(max_file_size_mb=2)

    # --- Included extensions ---

    @pytest.mark.parametrize("ext", [
        ".py", ".cpp", ".c", ".h", ".hpp", ".js", ".ts",
        ".java", ".go", ".rs", ".md", ".txt", ".yaml", ".yml",
        ".json", ".xml", ".toml", ".xacro", ".urdf",
    ])
    def test_included_extensions(self, ff, ext):
        include, reason = ff.should_include(f"src/main{ext}", 100)
        assert include is True, f"Expected {ext} to be included, but got: {reason}"

    # --- Included filenames ---

    @pytest.mark.parametrize("filename", [
        "CMakeLists.txt",
        "requirements.txt",
        "pyproject.toml",
        "package.json",
        "package.xml",
    ])
    def test_included_filenames(self, ff, filename):
        include, reason = ff.should_include(f"some/path/{filename}", 100)
        assert include is True, f"Expected {filename} to be included, but got: {reason}"

    # --- Included suffixes ---

    def test_launch_py_suffix(self, ff):
        include, reason = ff.should_include("robot/my_node.launch.py", 100)
        assert include is True

    # --- Non-included extensions ---

    @pytest.mark.parametrize("filename", [
        "image.png", "photo.jpg", "video.mp4", "program.exe",
        "library.dll", "archive.zip", "deps.lock",
    ])
    def test_excluded_extensions(self, ff, filename):
        include, reason = ff.should_include(f"assets/{filename}", 100)
        assert include is False

    def test_unknown_extension_excluded(self, ff):
        include, reason = ff.should_include("data/file.xyz", 100)
        assert include is False
        assert "not in include list" in reason


class TestFileFilterExcludedDirs:
    """Test directory exclusion rules."""

    @pytest.fixture
    def ff(self):
        return FileFilter(max_file_size_mb=2)

    @pytest.mark.parametrize("excluded_dir", [
        ".git", "node_modules", "venv", ".venv",
        "__pycache__", "build", "dist", "target", ".cache",
    ])
    def test_excluded_directories(self, ff, excluded_dir):
        include, reason = ff.should_include(f"{excluded_dir}/some_file.py", 100)
        assert include is False
        assert "excluded directory" in reason

    def test_nested_excluded_directory(self, ff):
        include, reason = ff.should_include("src/node_modules/package/index.js", 100)
        assert include is False

    def test_non_excluded_directory(self, ff):
        include, reason = ff.should_include("src/main/app.py", 100)
        assert include is True


class TestFileFilterSizeLimit:
    """Test file size enforcement."""

    def test_file_within_limit(self):
        ff = FileFilter(max_file_size_mb=2)
        include, reason = ff.should_include("main.py", 1_000_000)  # ~1MB
        assert include is True

    def test_file_exactly_at_limit(self):
        ff = FileFilter(max_file_size_mb=2)
        include, reason = ff.should_include("main.py", 2 * 1024 * 1024)  # exactly 2MB
        assert include is True

    def test_file_over_limit(self):
        ff = FileFilter(max_file_size_mb=2)
        include, reason = ff.should_include("huge.py", 3 * 1024 * 1024)  # 3MB
        assert include is False
        assert "too large" in reason

    def test_custom_size_limit(self):
        ff = FileFilter(max_file_size_mb=0.5)  # 512KB limit
        include, reason = ff.should_include("medium.py", 600_000)  # ~586KB
        assert include is False

    def test_env_max_file_size(self):
        """MAX_FILE_SIZE_MB from environment is respected."""
        with patch.dict("os.environ", {"MAX_FILE_SIZE_MB": "1"}):
            ff = FileFilter()
            assert ff.max_file_size_mb == 1.0
            include, reason = ff.should_include("big.py", 1_500_000)
            assert include is False

    def test_default_size_limit(self):
        """Default limit is 2MB when no env var is set."""
        with patch.dict("os.environ", {}, clear=True):
            ff = FileFilter()
            assert ff.max_file_size_mb == 2.0


class TestFileFilterEdgeCases:
    """Test edge cases in file filtering."""

    @pytest.fixture
    def ff(self):
        return FileFilter(max_file_size_mb=2)

    def test_windows_path_separators(self, ff):
        """Backslashes in paths should be handled correctly."""
        include, reason = ff.should_include("src\\main\\app.py", 100)
        assert include is True

    def test_windows_excluded_dir(self, ff):
        include, reason = ff.should_include("node_modules\\pkg\\index.js", 100)
        assert include is False

    def test_empty_filename(self, ff):
        include, reason = ff.should_include("", 0)
        assert include is False

    def test_dotfile(self, ff):
        """Dotfiles without known extensions should be excluded."""
        include, reason = ff.should_include(".gitignore", 100)
        assert include is False
