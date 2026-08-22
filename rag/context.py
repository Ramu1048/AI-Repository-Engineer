"""
RAG Context Builder.
Formats retrieved chunks with headers and enforces token/character budget constraints.
"""

from typing import List
from retrieval import RetrievedChunk


def build_context(
    retrieved_chunks: List[RetrievedChunk],
    max_chars: int = 15000,
) -> str:
    """
    Builds a structured context string from retrieved chunks.

    :param retrieved_chunks: List of RetrievedChunk objects from retrieval.
    :param max_chars: Maximum character budget for the context string.
    :return: Formatted context string with file, line, and symbol headers.
    """
    if not retrieved_chunks:
        return ""

    # Sort chunks by relevance score descending so highest quality chunks are retained
    sorted_chunks = sorted(retrieved_chunks, key=lambda c: getattr(c, "score", 0.0), reverse=True)

    formatted_blocks: List[str] = []
    current_char_count = 0

    for idx, r_chunk in enumerate(sorted_chunks, 1):
        chunk = r_chunk.chunk
        symbol_info = f" | Symbol: {chunk.symbol_name}" if getattr(chunk, "symbol_name", None) else ""
        header = (
            f"--- [CHUNK {idx}] File: {chunk.file_path} | "
            f"Lines: {chunk.start_line}-{chunk.end_line}{symbol_info} | "
            f"Score: {getattr(r_chunk, 'score', 1.0):.2f} ---"
        )
        block = f"{header}\n{chunk.content.strip()}\n"

        if current_char_count + len(block) > max_chars:
            # Token/character budget reached; omit remaining lower-scored chunks
            break

        formatted_blocks.append(block)
        current_char_count += len(block)

    return "\n\n".join(formatted_blocks)
