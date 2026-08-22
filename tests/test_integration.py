"""
End-to-end integration tests for Member 4 LLM & AI Analysis Module.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from analyzers.architecture import analyze as analyze_architecture
from analyzers.bugs import analyze as analyze_bugs
from analyzers.dependencies import analyze as analyze_dependencies
from analyzers.security import analyze as analyze_security
from github_analysis.issues import analyze_issue
from github_analysis.pull_requests import analyze_pr
from rag.chain import RAGAnswer, ask as ask_rag


class TestIntegration(unittest.TestCase):

    def test_end_to_end_module_workflow(self):
        """Validates full end-to-end workflow on a sample repository structure."""
        with tempfile.TemporaryDirectory() as repo_dir:
            # Create sample files
            main_py = os.path.join(repo_dir, "main.py")
            with open(main_py, "w", encoding="utf-8") as f:
                f.write(
                    'import os\n'
                    'import utils\n\n'
                    'API_SECRET = "sk-proj-1234567890abcdef"\n\n'
                    'def run_app(data=[]):\n'
                    '    try:\n'
                    '        val = 100 / 0\n'
                    '    except:\n'
                    '        pass\n'
                    '    return utils.process(val)\n'
                )

            utils_py = os.path.join(repo_dir, "utils.py")
            with open(utils_py, "w", encoding="utf-8") as f:
                f.write(
                    'def process(val):\n'
                    '    return val * 2\n'
                )

            req_txt = os.path.join(repo_dir, "requirements.txt")
            with open(req_txt, "w", encoding="utf-8") as f:
                f.write("requests>=2.28.0\npytest==7.1.2\n")

            # 1. Test Architecture Analysis
            arch_report = analyze_architecture(repo_dir)
            self.assertGreaterEqual(len(arch_report.modules), 2)
            self.assertIn("graph TD", arch_report.mermaid_diagram)

            # 2. Test Dependency Analysis
            deps = analyze_dependencies(repo_dir)
            self.assertEqual(len(deps), 2)
            self.assertTrue(any(d.name == "requests" for d in deps))

            # 3. Test Bug Analysis
            bug_findings = analyze_bugs(repo_dir)
            self.assertGreaterEqual(len(bug_findings), 2)
            codes = [b.issue_code for b in bug_findings]
            self.assertIn("B006", codes)   # Mutable default
            self.assertIn("ZE001", codes)  # Zero division

            # 4. Test Security Analysis & Masking
            sec_findings = analyze_security(repo_dir)
            self.assertGreaterEqual(len(sec_findings), 1)
            for sf in sec_findings:
                self.assertNotIn("sk-proj-1234567890abcdef", sf.masked_value)

            # 5. Test RAG Ask with Mock LLM and Mock Retrieval
            from retrieval import Chunk, RetrievedChunk
            fake_chunks = [
                RetrievedChunk(
                    chunk=Chunk(
                        repository_id=repo_dir,
                        content="def run_app(data=[]):\n    try:\n        val = 100 / 0\n    except:\n        pass\n    return utils.process(val)\n",
                        file_path="main.py",
                        language="python",
                        symbol_type="function",
                        symbol_name="run_app",
                        class_name=None,
                        start_line=1,
                        end_line=12,
                    ),
                    score=0.95,
                    retrieval_method="vector",
                )
            ]

            mock_provider = MagicMock()
            mock_provider.generate.return_value = (
                "Based on [main.py:L1-L12], `run_app` calls `utils.process` after calculating `val`."
            )

            with patch("rag.chain.retrieve", return_value=fake_chunks):
                with patch("rag.chain.get_llm_provider", return_value=mock_provider):
                    rag_ans = ask_rag(repo_dir, "What does run_app do?")
                    self.assertIsInstance(rag_ans, RAGAnswer)
                    self.assertIn("run_app", rag_ans.answer)
                    self.assertGreater(len(rag_ans.sources), 0)

            # 6. Test GitHub Issue and PR Analysis
            with patch("github_analysis.issues.ask_rag", return_value=rag_ans):
                issue_res = analyze_issue(repo_dir, 1)
                self.assertEqual(issue_res.issue_number, 1)
                self.assertIn(issue_res.confidence, ("likely", "potential"))

            with patch("github_analysis.pull_requests.get_llm_provider", return_value=mock_provider):
                pr_res = analyze_pr(repo_dir, 5)
                self.assertEqual(pr_res.pr_number, 5)
                self.assertEqual(pr_res.code_quality_score, "Good")


if __name__ == "__main__":
    unittest.main()
