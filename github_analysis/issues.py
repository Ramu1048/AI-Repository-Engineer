"""
GitHub Issue Analysis Module.
Fetches GitHub issue data and invokes RAG chain to identify affected files and root cause hypotheses.
"""

from dataclasses import dataclass, field
import re
from typing import List, Optional

from github_service import GitHubClient
from rag.chain import ask as ask_rag


@dataclass
class IssueAnalysis:
    issue_number: int
    title: str
    status: str
    likely_affected_files: List[str] = field(default_factory=list)
    root_cause_hypothesis: str = ""
    related_components: List[str] = field(default_factory=list)
    summary: str = ""
    confidence: str = "likely"
    suggested_fix: str = ""


def analyze_issue(repository_id: str, issue_number: int) -> IssueAnalysis:
    """
    Analyzes a GitHub issue by combining issue metadata with RAG context retrieval.

    :param repository_id: Repository identifier or directory path.
    :param issue_number: GitHub issue number.
    :return: IssueAnalysis dataclass.
    """
    gh_client = GitHubClient()
    issue_data = gh_client.get_issue(repository_id, issue_number)

    if not issue_data:
        raise ValueError(f"Issue #{issue_number} not found for repository '{repository_id}'.")

    title = issue_data.get("title", f"Issue #{issue_number}")
    body = issue_data.get("body", "")
    comments = issue_data.get("comments", [])
    comment_text = "\n".join([f"- {c.get('author', 'user')}: {c.get('body', '')}" for c in comments])

    # Construct RAG prompt query from issue text
    rag_query = f"Analyze Issue #{issue_number}: {title}\nDescription: {body}\nComments:\n{comment_text}"

    # Call RAG chain
    rag_response = ask_rag(repository_id, rag_query)

    # Extract source files
    affected_files = [src.file_path for src in rag_response.sources]

    # Deduplicate and extract components
    components = sorted(list({f.split("/")[0] for f in affected_files if "/" in f}))

    summary = f"Analysis of Issue #{issue_number} ({title}). Identified {len(affected_files)} potential source references."

    return IssueAnalysis(
        issue_number=issue_number,
        title=title,
        status=issue_data.get("state", "open"),
        likely_affected_files=affected_files,
        root_cause_hypothesis=rag_response.answer,
        related_components=components,
        summary=summary,
        confidence="likely" if affected_files else "potential",
        suggested_fix="Review highlighted source references and apply targeted exception handling/validation.",
    )

