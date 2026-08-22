"""
RAG Package for AI Repository Engineer.
Exposes ask(), RAGAnswer, SourceRef, build_context, and system prompts.
"""

from rag.chain import RAGAnswer, SourceRef, ask
from rag.context import build_context
from rag.prompt import get_rag_prompt

__all__ = ["ask", "RAGAnswer", "SourceRef", "build_context", "get_rag_prompt"]
