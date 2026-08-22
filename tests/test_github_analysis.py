"""
Unit tests for GitHub issue, PR, and commit analysis functions.
"""

import unittest
from unittest.mock import MagicMock, patch

from github_analysis import (
    CommitAnalysis,
    IssueAnalysis,
    PRReview,
    analyze_commit,
    analyze_issue,
    analyze_pr,
)


class TestGitHubAnalysis(unittest.TestCase):

    def test_analyze_issue(self):
        mock_rag_res = MagicMock()
        mock_rag_res.answer = "Hypothesized root cause in auth context handler."
        mock_rag_res.sources = [MagicMock(file_path="auth/service.py")]

        with patch("github_analysis.issues.ask_rag", return_value=mock_rag_res):
            res = analyze_issue("repo1", 42)
            self.assertIsInstance(res, IssueAnalysis)
            self.assertEqual(res.issue_number, 42)
            self.assertIn("auth/service.py", res.likely_affected_files)
            self.assertEqual(res.confidence, "likely")

    def test_analyze_pr(self):
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "PR looks high quality with low risk."

        with patch("github_analysis.pull_requests.get_llm_provider", return_value=mock_provider):
            res = analyze_pr("repo1", 101)
            self.assertIsInstance(res, PRReview)
            self.assertEqual(res.pr_number, 101)
            self.assertEqual(res.code_quality_score, "Good")

    def test_analyze_pr_diff_truncation(self):
        huge_diff = "+" + ("line\n+" * 10000)
        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = {
            "title": "Big PR",
            "diff": huge_diff,
            "changed_files": ["big_file.py"],
        }
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "Reviewed truncated diff successfully."

        with patch("github_analysis.pull_requests.GitHubClient", return_value=mock_gh):
            with patch("github_analysis.pull_requests.get_llm_provider", return_value=mock_provider):
                res = analyze_pr("repo1", 999)
                self.assertIsInstance(res, PRReview)
                prompt_arg = mock_provider.generate.call_args[1]["user_prompt"]
                self.assertIn("[Diff content truncated due to length limit]", prompt_arg)

    def test_analyze_missing_pr(self):
        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = None

        with patch("github_analysis.pull_requests.GitHubClient", return_value=mock_gh):
            with self.assertRaises(ValueError):
                analyze_pr("repo1", 404)

    def test_analyze_commit(self):
        res = analyze_commit("repo1", "abc123def456")
        self.assertIsInstance(res, CommitAnalysis)
        self.assertEqual(res.commit_hash, "abc123def456")
        self.assertTrue(res.needs_reindex)


if __name__ == "__main__":
    unittest.main()

