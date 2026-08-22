"""
File filtering rules for repository ingestion.

Configuration is exposed as module-level constants so other members/config
can extend them. The FileFilter class applies these rules to determine
whether a file should be included in the index.
"""

import os
import logging

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — extend these sets to add new supported file types
# ---------------------------------------------------------------------------

# File extensions to include (lowercase, with leading dot)
INCLUDED_EXTENSIONS: set[str] = {
    # Python
    ".py",
    # C / C++
    ".cpp", ".c", ".h", ".hpp",
    # JavaScript / TypeScript
    ".js", ".ts",
    # Java
    ".java",
    # Go
    ".go",
    # Rust
    ".rs",
    # Documentation / Data
    ".md", ".txt", ".yaml", ".yml", ".json", ".xml", ".toml",
    # Robotics / ROS
    ".xacro", ".urdf",
}

# Exact filenames to include (regardless of extension)
INCLUDED_FILENAMES: set[str] = {
    "CMakeLists.txt",
    "requirements.txt",
    "pyproject.toml",
    "package.json",
    "package.xml",
    "README",
    "LICENSE",
    "Makefile",
    "Dockerfile",
}

# Filename suffixes to include (e.g. foo.launch.py)
INCLUDED_SUFFIXES: list[str] = [
    ".launch.py",
]

# Directory names to exclude (matched against any path component)
EXCLUDED_DIRS: set[str] = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "target",
    ".cache",
}

# File extensions to always exclude (binary / generated)
EXCLUDED_EXTENSIONS: set[str] = {
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".bmp", ".webp",
    # Video / Audio
    ".mp4", ".avi", ".mov", ".mp3", ".wav",
    # Executables / Libraries
    ".exe", ".dll", ".so", ".o", ".a", ".dylib",
    # Python compiled
    ".pyc", ".pyo", ".pyd",
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    # Package artifacts
    ".whl", ".egg",
    # Lock files
    ".lock",
}

# Default maximum file size in MB
DEFAULT_MAX_FILE_SIZE_MB = 2


class FileFilter:
    """
    Determines whether a file should be included in the repository index.

    Checks (in order):
    1. Is the file in an excluded directory?
    2. Does the file have an excluded extension?
    3. Does the file have an included extension or filename?
    4. Is the file within the size limit?
    """

    def __init__(self, max_file_size_mb: float | None = None):
        """
        Initialize the file filter.

        Args:
            max_file_size_mb: Maximum file size in MB. Falls back to
                MAX_FILE_SIZE_MB env var, then DEFAULT_MAX_FILE_SIZE_MB.
        """
        if max_file_size_mb is not None:
            self.max_file_size_mb = max_file_size_mb
        else:
            env_val = os.getenv("MAX_FILE_SIZE_MB", "").strip()
            if env_val:
                try:
                    self.max_file_size_mb = float(env_val)
                except ValueError:
                    logger.warning(
                        f"Invalid MAX_FILE_SIZE_MB value: '{env_val}', "
                        f"using default {DEFAULT_MAX_FILE_SIZE_MB}MB"
                    )
                    self.max_file_size_mb = DEFAULT_MAX_FILE_SIZE_MB
            else:
                self.max_file_size_mb = DEFAULT_MAX_FILE_SIZE_MB

        self.max_file_size_bytes = int(self.max_file_size_mb * 1024 * 1024)

    def should_include(self, file_path: str, file_size_bytes: int) -> tuple[bool, str | None]:
        """
        Determine whether a file should be included in the index.

        Args:
            file_path: Relative or absolute file path.
            file_size_bytes: Size of the file in bytes.

        Returns:
            Tuple of (should_include, skip_reason).
            If should_include is True, skip_reason is None.
            If should_include is False, skip_reason explains why.
        """
        # Normalize path separators
        normalized = file_path.replace("\\", "/")
        parts = normalized.split("/")
        basename = parts[-1] if parts else ""

        # 1. Check excluded directories
        for part in parts[:-1]:  # All path components except the filename
            if part in EXCLUDED_DIRS:
                return False, f"in excluded directory: {part}"

        # 2. Check excluded extensions
        _, ext = os.path.splitext(basename)
        ext_lower = ext.lower()
        if ext_lower in EXCLUDED_EXTENSIONS:
            return False, f"excluded extension: {ext_lower}"

        # 3. Check included filenames (exact match)
        if basename in INCLUDED_FILENAMES:
            # Passes name check, now check size
            return self._check_size(file_size_bytes, basename)

        # 4. Check included suffixes (e.g. .launch.py)
        for suffix in INCLUDED_SUFFIXES:
            if basename.endswith(suffix):
                return self._check_size(file_size_bytes, basename)

        # 5. Check included extensions
        if ext_lower in INCLUDED_EXTENSIONS:
            return self._check_size(file_size_bytes, basename)

        # Not in any include list
        return False, f"extension '{ext_lower}' not in include list"

    def _check_size(self, file_size_bytes: int, basename: str) -> tuple[bool, str | None]:
        """Check if file is within the size limit."""
        if file_size_bytes > self.max_file_size_bytes:
            size_mb = file_size_bytes / (1024 * 1024)
            return False, (
                f"file too large: {basename} is {size_mb:.2f}MB "
                f"(limit: {self.max_file_size_mb}MB)"
            )
        return True, None
