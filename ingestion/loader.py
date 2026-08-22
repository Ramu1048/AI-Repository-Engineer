"""
Repository file loader — walks a cloned repo, applies filtering, and reads text.

This is the handoff point to Member 2 (Code Intelligence). The LoadedFile
dataclass and load_repository_files() function are the contract interface.
"""

import os
import logging
from dataclasses import dataclass

from ingestion.file_filter import FileFilter, EXCLUDED_DIRS
from exceptions import EmptyRepositoryError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extension → language mapping
# ---------------------------------------------------------------------------

EXTENSION_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",       # Could be C or C++; default to C
    ".hpp": "cpp",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".md": "markdown",
    ".txt": "text",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".xml": "xml",
    ".toml": "toml",
    ".xacro": "xml",
    ".urdf": "xml",
}


@dataclass
class LoadedFile:
    """
    A filtered, decoded text file ready for parsing.

    This is the contract interface that Member 2's parser consumes.
    Do not change this shape without coordinating with the team.
    """
    file_path: str      # relative path within the repo
    content: str         # full text content (UTF-8 decoded)
    language: str        # detected from extension
    size_bytes: int


FILENAME_LANGUAGE_MAP: dict[str, str] = {
    "README": "markdown",
    "LICENSE": "text",
    "Makefile": "makefile",
    "Dockerfile": "dockerfile",
    "CMakeLists.txt": "cmake",
    "requirements.txt": "text",
    "pyproject.toml": "toml",
    "package.json": "json",
    "package.xml": "xml",
}


def detect_language(file_path: str) -> str:
    """
    Detect the programming language from a file's extension or exact filename.

    Args:
        file_path: Filename or path.

    Returns:
        Language name string (lowercase). Returns 'unknown' if not mapped.
    """
    basename = os.path.basename(file_path)

    # Check exact filename map first
    if basename in FILENAME_LANGUAGE_MAP:
        return FILENAME_LANGUAGE_MAP[basename]

    # Check compound suffixes (e.g. .launch.py)
    if basename.endswith(".launch.py"):
        return "python"

    _, ext = os.path.splitext(basename)
    return EXTENSION_LANGUAGE_MAP.get(ext.lower(), "unknown")


def detect_primary_languages(files: list[LoadedFile]) -> list[str]:
    """
    Detect the repository's primary language(s) by file extension distribution.

    Returns languages sorted by file count (descending), excluding
    non-code types like 'text', 'markdown', 'json', etc.

    Args:
        files: List of LoadedFile objects.

    Returns:
        List of language names sorted by frequency.
    """
    # Non-code types to exclude from "primary language" detection
    non_code_types = {"text", "markdown", "json", "yaml", "toml", "xml", "unknown"}

    language_counts: dict[str, int] = {}
    for f in files:
        lang = f.language
        if lang not in non_code_types:
            language_counts[lang] = language_counts.get(lang, 0) + 1

    if not language_counts:
        # Fall back to all languages if no code files found
        for f in files:
            language_counts[f.language] = language_counts.get(f.language, 0) + 1

    # Sort by count descending
    sorted_langs = sorted(language_counts.items(), key=lambda x: x[1], reverse=True)
    return [lang for lang, _ in sorted_langs]


def load_repository_files(
    repository_id: str,
    base_dir: str = "data/repositories",
) -> list[LoadedFile]:
    """
    Walk a cloned repository, filter files, and read them as UTF-8 text.

    This is the main entry point that Member 2's parser calls.

    Args:
        repository_id: Deterministic repo ID (used as directory name).
        base_dir: Parent directory containing all cloned repos.

    Returns:
        List of LoadedFile objects — filtered, decoded, ready for parsing.

    Raises:
        FileNotFoundError: Clone directory doesn't exist.
        EmptyRepositoryError: Zero indexable files after filtering.
    """
    repo_dir = os.path.abspath(os.path.join(base_dir, repository_id))

    if not os.path.isdir(repo_dir):
        raise FileNotFoundError(
            f"Repository directory not found: {repo_dir}. "
            f"Was the repository cloned successfully?"
        )

    file_filter = FileFilter()
    loaded_files: list[LoadedFile] = []
    total_files = 0
    skipped_files = 0
    skip_reasons: dict[str, int] = {}

    for root, dirs, files in os.walk(repo_dir):
        # Prune excluded directories in-place for efficiency
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

        for filename in files:
            total_files += 1
            abs_path = os.path.join(root, filename)

            # Get relative path from repo root
            rel_path = os.path.relpath(abs_path, repo_dir)
            rel_path = rel_path.replace("\\", "/")  # Normalize to forward slashes

            # Get file size
            try:
                file_size = os.path.getsize(abs_path)
            except OSError as e:
                logger.warning(f"Cannot stat file {rel_path}: {e}")
                skipped_files += 1
                skip_reasons["stat_error"] = skip_reasons.get("stat_error", 0) + 1
                continue

            # Apply filter
            include, reason = file_filter.should_include(rel_path, file_size)
            if not include:
                skipped_files += 1
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                logger.debug(f"Skipped: {rel_path} — {reason}")
                continue

            # Read file content as UTF-8
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError as e:
                skipped_files += 1
                reason = "decode_error"
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                logger.warning(f"UTF-8 decode failed for {rel_path}: {e}")
                continue
            except OSError as e:
                skipped_files += 1
                reason = "read_error"
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                logger.warning(f"Cannot read file {rel_path}: {e}")
                continue

            language = detect_language(rel_path)
            loaded_files.append(LoadedFile(
                file_path=rel_path,
                content=content,
                language=language,
                size_bytes=file_size,
            ))

    # Summary logging
    logger.info(
        f"Loading complete: {len(loaded_files)} files indexed, "
        f"{skipped_files} skipped, {total_files} total"
    )
    if skip_reasons:
        for reason, count in sorted(skip_reasons.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  Skip reason: {reason} ({count} files)")

    if not loaded_files:
        # Parse owner/repo from repository_id for the error message
        parts = repository_id.rsplit("_", 2)
        owner = parts[0] if len(parts) >= 3 else "unknown"
        repo = parts[1] if len(parts) >= 3 else repository_id
        raise EmptyRepositoryError(owner, repo)

    return loaded_files
