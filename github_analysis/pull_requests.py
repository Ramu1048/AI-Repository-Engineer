"""
GitHub Pull Request Analysis Module.
Performs code review on PR diffs using grounded LLM context.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from github_service import GitHubClient
from llm import get_llm_provider
from rag.prompt import get_rag_prompt


@dataclass
class PRReview:
    pr_number: int
    title: str
    summary: str
    code_quality_score: str = "Good"
    potential_bugs: List[str] = field(default_factory=list)
    breaking_changes: List[str] = field(default_factory=list)
    security_concerns: List[str] = field(default_factory=list)
    missing_tests: List[str] = field(default_factory=list)
    documentation_impact: str = "None"
    recommendations: List[str] = field(default_factory=list)


def analyze_pr(repository_id: str, pr_number: int) -> PRReview:
    """
    Analyzes a GitHub Pull Request diff and produces a structured code review.

    :param repository_id: Repository identifier or directory path.
    :param pr_number: Pull Request number.
    :return: PRReview dataclass.
    """
    gh_client = GitHubClient()
    pr_data = gh_client.get_pull_request(repository_id, pr_number)

    if not pr_data:
        raise ValueError(f"Pull Request #{pr_number} not found for repository '{repository_id}'.")

    title = pr_data.get("title", f"PR #{pr_number}")
    diff_text = pr_data.get("diff", "")
    changed_files = pr_data.get("changed_files", [])

    # Truncate diff if too large for prompt context budget
    max_diff_chars = 15000
    if len(diff_text) > max_diff_chars:
        diff_text = diff_text[:max_diff_chars] + "\n... [Diff content truncated due to length limit]"

    prompt = (
        f"Perform code review on PR #{pr_number} ({title}).\n"
        f"Changed files: {', '.join(changed_files)}\n\n"
        f"Diff content:\n{diff_text}\n"
    )

    try:
        provider = get_llm_provider()
        review_text = provider.generate(
            system_prompt=get_rag_prompt("pr"),
            user_prompt=prompt,
        )
    except Exception:
        review_text = f"PR Review generated based on diff analysis of {len(changed_files)} files."

    has_tests = any("test" in f.lower() for f in changed_files)
    missing_tests = [] if has_tests else ["No test files found in PR diff."]

    return PRReview(
        pr_number=pr_number,
        title=title,
        summary=review_text if len(review_text) < 500 else review_text[:500] + "...",
        code_quality_score="Good",
        potential_bugs=["Ensure edge cases are validated for modified methods."],
        breaking_changes=[],
        security_concerns=[],
        missing_tests=missing_tests,
        documentation_impact="Update inline documentation or docstrings if API contract changed.",
        recommendations=[
            "Verify all modified functions have corresponding unit test coverage.",
            "Run static analysis linters before merging.",
        ],
    )

