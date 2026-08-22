"""
Unit tests for RAG chain, context builder, prompt templates, and source extraction.
"""

import unittest
from unittest.mock import MagicMock, patch

from rag.chain import RAGAnswer, SourceRef, ask
from rag.context import build_context
from rag.prompt import get_rag_prompt
from retrieval import Chunk, RetrievedChunk


class TestRAG(unittest.TestCase):

    def test_build_context_headers(self):
        chunks = [
            RetrievedChunk(
                chunk=Chunk(
                    repository_id="test_repo",
                    content="def hello(): return 'world'",
                    file_path="src/main.py",
                    language="python",
                    symbol_type="function",
                    symbol_name="hello",
                    class_name=None,
                    start_line=10,
                    end_line=12,
                ),
                score=0.95,
                retrieval_method="vector",
            )
        ]

        context_str = build_context(chunks)
        self.assertIn("--- [CHUNK 1] File: src/main.py | Lines: 10-12 | Symbol: hello | Score: 0.95 ---", context_str)
        self.assertIn("def hello(): return 'world'", context_str)

    def test_build_context_budget_truncation(self):
        chunks = [
            RetrievedChunk(
                chunk=Chunk(repository_id="test_repo", content="A" * 1000, file_path="file1.py", language="python", symbol_type="function", symbol_name="funcA", class_name=None, start_line=1, end_line=10),
                score=0.9,
                retrieval_method="vector",
            ),
            RetrievedChunk(
                chunk=Chunk(repository_id="test_repo", content="B" * 1000, file_path="file2.py", language="python", symbol_type="function", symbol_name="funcB", class_name=None, start_line=1, end_line=10),
                score=0.5,
                retrieval_method="vector",
            ),
        ]

        # Limit budget to 1200 characters, so second chunk is omitted
        context_str = build_context(chunks, max_chars=1200)
        self.assertIn("file1.py", context_str)
        self.assertNotIn("file2.py", context_str)

    def test_ask_empty_retrieval(self):
        with patch("rag.chain.retrieve", return_value=[]):
            res = ask("non_existent_repo", "Where is the authentication handler?")
            self.assertIsInstance(res, RAGAnswer)
            self.assertIn("INSUFFICIENT CONTEXT", res.answer)
            self.assertEqual(len(res.sources), 0)
            self.assertEqual(res.confidence, "none")

    def test_ask_grounding_and_sources(self):
        fake_chunks = [
            RetrievedChunk(
                chunk=Chunk(
                    repository_id="test_repo",
                    content="class Auth:\n    def login(self): pass",
                    file_path="auth/service.py",
                    language="python",
                    symbol_type="class",
                    symbol_name="Auth.login",
                    class_name="Auth",
                    start_line=1,
                    end_line=5,
                ),
                score=0.98,
                retrieval_method="vector",
            )
        ]

        mock_provider = MagicMock()
        mock_provider.generate.return_value = "The Auth class is defined in [auth/service.py:L1-L5]."

        with patch("rag.chain.retrieve", return_value=fake_chunks):
            with patch("rag.chain.get_llm_provider", return_value=mock_provider):
                res = ask("test_repo", "Where is the login function?")

                self.assertIsInstance(res, RAGAnswer)
                self.assertIn("Auth class", res.answer)
                self.assertEqual(len(res.sources), 1)
                self.assertEqual(res.sources[0].file_path, "auth/service.py")
                self.assertEqual(res.sources[0].start_line, 1)
                self.assertEqual(res.sources[0].end_line, 5)

    def test_ask_unanswerable_grounding(self):
        fake_chunks = [
            RetrievedChunk(
                chunk=Chunk(repository_id="test_repo", content="def foo(): pass", file_path="foo.py", language="python", symbol_type="function", symbol_name="foo", class_name=None, start_line=1, end_line=2),
                score=0.3,
                retrieval_method="vector",
            )
        ]
        mock_provider = MagicMock()
        mock_provider.generate.return_value = (
            "INSUFFICIENT CONTEXT: The retrieved code snippets do not contain enough information to answer database query questions."
        )

        with patch("rag.chain.retrieve", return_value=fake_chunks):
            with patch("rag.chain.get_llm_provider", return_value=mock_provider):
                res = ask("test_repo", "How is postgres connected?")

                self.assertIn("INSUFFICIENT CONTEXT", res.answer)
                self.assertEqual(res.confidence, "low")


if __name__ == "__main__":
    unittest.main()

