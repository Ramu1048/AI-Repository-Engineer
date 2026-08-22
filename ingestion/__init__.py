"""
Ingestion package — file filtering, loading, and metadata building.
"""

from ingestion.loader import LoadedFile, load_repository_files
from ingestion.metadata import RepositoryMetadata, build_metadata
from ingestion.file_filter import FileFilter
from ingestion.chunker import chunk_repository

__all__ = [
    "LoadedFile",
    "load_repository_files",
    "RepositoryMetadata",
    "build_metadata",
    "FileFilter",
    "chunk_repository",
]

