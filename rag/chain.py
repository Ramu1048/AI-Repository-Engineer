"""
RAG Chain Orchestrator.
Orchestrates retrieve -> context building -> LLM generation -> source extraction.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from llm import get_llm_provider, LLMProviderError
from rag.context import build_context
from rag.prompt import get_rag_prompt
from retrieval import retrieve, RetrievedChunk


@dataclass
class SourceRef:
    file_path: str
    start_line: int
    end_line: int
    symbol_name: Optional[str] = None

    def __hash__(self):
        return hash((self.file_path, self.start_line, self.end_line, self.symbol_name))

    def __eq__(self, other):
        if not isinstance(other, SourceRef):
            return False
        return (
            self.file_path == other.file_path
            and self.start_line == other.start_line
            and self.end_line == other.end_line
            and self.symbol_name == other.symbol_name
        )


@dataclass
class RAGAnswer:
    answer: str
    sources: List[SourceRef] = field(default_factory=list)
    context_used: str = ""
    confidence: str = "high"


def ask(
    repository_id: str,
    question: str,
    top_k: int = 8,
    llm_provider_name: Optional[str] = None,
) -> RAGAnswer:
    """
    Executes the full RAG pipeline:
    1. Call retrieval to find top_k relevant code chunks.
    2. Format retrieved chunks into grounded context string.
    3. Generate grounded answer using active LLMProvider.
    4. Extract deduplicated source citations.
    """
    # 1. Retrieve chunks from Member 3's retrieval engine
    retrieved_chunks: List[RetrievedChunk] = retrieve(repository_id, question, top_k=top_k)

    # 2. Check for empty context
    if not retrieved_chunks:
        return RAGAnswer(
            answer=(
                "INSUFFICIENT CONTEXT: No relevant code snippets could be retrieved "
                f"from repository '{repository_id}' for question: '{question}'."
            ),
            sources=[],
            context_used="",
            confidence="none",
        )

    # 3. Build context
    context_str = build_context(retrieved_chunks)

    # 4. Prepare system prompt & LLM provider
    system_prompt = get_rag_prompt("rag")
    provider = get_llm_provider(llm_provider_name)

    # 5. Call LLM
    try:
        raw_answer = provider.generate(
            system_prompt=system_prompt,
            user_prompt=question,
            context=context_str,
        )
    except LLMProviderError as e:
        # Re-raise or return structured failure notice
        raise e

    # 6. Extract deduplicated sources in citation order
    sources: List[SourceRef] = []
    seen_sources = set()

    for r_chunk in retrieved_chunks:
        c = r_chunk.chunk
        s_ref = SourceRef(
            file_path=c.file_path,
            start_line=c.start_line,
            end_line=c.end_line,
            symbol_name=getattr(c, "symbol_name", None),
        )
        if s_ref not in seen_sources:
            seen_sources.add(s_ref)
            sources.append(s_ref)

    # Determine confidence level
    confidence = "high"
    if "INSUFFICIENT CONTEXT" in raw_answer.upper():
        confidence = "low"

    return RAGAnswer(
        answer=raw_answer,
        sources=sources,
        context_used=context_str,
        confidence=confidence,
    )
