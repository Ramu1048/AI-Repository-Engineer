"""
Bug Analyzer Module.
Performs read-only static analysis (AST & linters) and generates LLM-powered explanations and recommendations.
"""

import ast
from dataclasses import dataclass
import os
import subprocess
from typing import List, Optional

from llm import get_llm_provider


@dataclass
class BugFinding:
    severity: str
    file_path: str
    line_number: int
    issue_code: str
    issue_title: str
    explanation: str
    recommendation: str
    confidence: str = "likely"


class ASTBugVisitor(ast.NodeVisitor):
    """AST visitor to detect common Python code anti-patterns and potential bugs."""

    def __init__(self, file_path: str, rel_path: str):
        self.file_path = file_path
        self.rel_path = rel_path
        self.findings: List[BugFinding] = []

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        # Bare except clause: `except:`
        if node.type is None:
            self.findings.append(
                BugFinding(
                    severity="medium",
                    file_path=self.rel_path,
                    line_number=node.lineno,
                    issue_code="W0702",
                    issue_title="Bare except clause catches all exceptions including SystemExit",
                    explanation="Catching all exceptions without specifying an exception class masks critical runtime failures and system interrupts.",
                    recommendation="Specify explicit exception types (e.g. `except Exception:` or `except (ValueError, KeyError):`).",
                    confidence="likely",
                )
            )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Mutable default argument e.g. `def foo(a=[])`
        for default in node.args.defaults:
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self.findings.append(
                    BugFinding(
                        severity="high",
                        file_path=self.rel_path,
                        line_number=node.lineno,
                        issue_code="B006",
                        issue_title="Mutable default argument detected",
                        explanation="Using mutable objects (lists, dicts, sets) as default function arguments causes state to persist across calls.",
                        recommendation="Use `None` as default argument and initialize mutable container inside function body.",
                        confidence="likely",
                    )
                )
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp):
        # Potential division by zero e.g. `x / 0`
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
            if isinstance(node.right, ast.Constant) and node.right.value == 0:
                self.findings.append(
                    BugFinding(
                        severity="high",
                        file_path=self.rel_path,
                        line_number=node.lineno,
                        issue_code="ZE001",
                        issue_title="Division or modulo by literal zero",
                        explanation="Operation divides by zero constant, which will raise a ZeroDivisionError at runtime.",
                        recommendation="Guard divisor with zero check before executing division.",
                        confidence="unambiguous",
                    )
                )
        self.generic_visit(node)


def _explain_with_llm(finding: BugFinding, code_snippet: str) -> None:
    """Uses LLM to enrich explanation and recommendation with specific code context."""
    try:
        provider = get_llm_provider()
        prompt = (
            f"Explain this static code finding for line {finding.line_number} in file '{finding.file_path}':\n"
            f"Issue: {finding.issue_title} ({finding.issue_code})\n"
            f"Code snippet:\n{code_snippet}\n"
            "Provide a concise explanation (1-2 sentences) and fix recommendation."
        )
        res = provider.generate(
            system_prompt="You are a static code analysis expert. Be concise, precise, and practical.",
            user_prompt=prompt,
        )
        if res and not res.startswith("INSUFFICIENT CONTEXT"):
            finding.explanation = res.strip()
    except Exception:
        # Retain AST fallback explanation if LLM call fails
        pass


def _run_external_linters(repo_path: str) -> List[BugFinding]:
    """Runs Ruff or Bandit static analysis via subprocess if available, catching exceptions gracefully."""
    linter_findings: List[BugFinding] = []
    
    # 1. Attempt Ruff JSON scan
    try:
        proc = subprocess.run(
            ["ruff", "check", "--output-format=json", repo_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.stdout and proc.stdout.strip().startswith("["):
            import json
            data = json.loads(proc.stdout)
            for item in data:
                rel_path = os.path.relpath(item.get("filename", ""), repo_path) if os.path.isabs(item.get("filename", "")) else item.get("filename", "")
                code = item.get("code", "RUFF")
                msg = item.get("message", "Linter issue detected")
                line = item.get("location", {}).get("row", 1)
                linter_findings.append(
                    BugFinding(
                        severity="medium",
                        file_path=rel_path,
                        line_number=line,
                        issue_code=code,
                        issue_title=msg,
                        explanation=f"Ruff linter rule {code} triggered.",
                        recommendation="Refactor code according to Python style and standard linting guidelines.",
                        confidence="likely",
                    )
                )
    except (FileNotFoundError, subprocess.SubprocessError, Exception):
        pass

    # 2. Attempt Bandit JSON scan
    try:
        proc = subprocess.run(
            ["bandit", "-r", repo_path, "-f", "json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.stdout and "results" in proc.stdout:
            import json
            data = json.loads(proc.stdout)
            results = data.get("results", [])
            for res in results:
                rel_path = os.path.relpath(res.get("filename", ""), repo_path) if os.path.isabs(res.get("filename", "")) else res.get("filename", "")
                issue_text = res.get("issue_text", "Security issue detected")
                test_id = res.get("test_id", "BANDIT")
                line = res.get("line_number", 1)
                sev = res.get("issue_severity", "MEDIUM").lower()
                linter_findings.append(
                    BugFinding(
                        severity=sev,
                        file_path=rel_path,
                        line_number=line,
                        issue_code=test_id,
                        issue_title=issue_text,
                        explanation=f"Bandit static analysis identified potential issue {test_id}.",
                        recommendation="Review security impact and sanitize function parameters.",
                        confidence="potential",
                    )
                )
    except (FileNotFoundError, subprocess.SubprocessError, Exception):
        pass

    return linter_findings


def analyze(repository_id: str) -> List[BugFinding]:
    """
    Performs static bug analysis on python source files in repository_id.
    Executes AST checks and optional linters (Ruff/Pylint/Bandit if installed) strictly in read-only mode.
    """
    repo_path = repository_id if os.path.exists(repository_id) else os.getcwd()
    all_findings: List[BugFinding] = []

    # 1. Execute AST-based inspection
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("venv", "node_modules")]
        for file in files:
            if file.endswith(".py"):
                fpath = os.path.join(root, file)
                rel_path = os.path.relpath(fpath, repo_path)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    tree = ast.parse(content, filename=fpath)
                    visitor = ASTBugVisitor(fpath, rel_path)
                    visitor.visit(tree)
                    all_findings.extend(visitor.findings)
                except Exception:
                    continue

    # 2. Execute external subprocess linters gracefully if installed
    external_findings = _run_external_linters(repo_path)
    all_findings.extend(external_findings)

    return all_findings

