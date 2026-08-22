"""
Security Analyzer Module.
Scans repository code for hardcoded secrets, tokens, API keys, and unsafe function usage.
Applies mandatory secret masking on all outputs to prevent credential leaks.
"""

import ast
from dataclasses import dataclass
import os
import re
from typing import List


@dataclass
class SecurityFinding:
    file_path: str
    line_number: int
    category: str
    masked_value: str
    recommendation: str
    severity: str = "high"
    confidence: str = "likely"


def mask_secret(secret_val: str) -> str:
    """
    Masks secret strings so no raw keys are ever returned to UI, logs, or LLMs.
    Example: 'sk-1234567890abcdef' -> 'sk****ef'
    """
    if not secret_val:
        return ""
    val = secret_val.strip("'\" \t")
    if len(val) <= 4:
        return "****"
    elif len(val) <= 8:
        return f"{val[:1]}****{val[-1:]}"
    else:
        return f"{val[:2]}****{val[-2:]}"


# Regex patterns for detecting sensitive tokens and secrets
SECRET_PATTERNS = [
    (r"AKIA[0-9A-Z]{16}", "AWS Access Key ID"),
    (r"ghp_[0-9a-zA-Z]{36}", "GitHub Personal Access Token"),
    (r"xox[baprs]-[0-9a-zA-Z]{10,}", "Slack API Token"),
    (r"sk-[a-zA-Z0-9_-]{20,}", "API Key / Service Token"),
    (r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----", "Private RSA Key Header"),
    (
        r"(?i)(api[_-]?(key|secret)|secret[_-]?(key|token)?|password|auth[_-]?token|access[_-]?token|app[_-]?secret)\s*=\s*['\"]([^'\"]{6,})['\"]",
        "Hardcoded Credential / API Key",
    ),
]

UNSAFE_FUNCTION_NAMES = {
    "eval": ("Dynamic Code Evaluation", "Use ast.literal_eval or structured parser instead of eval."),
    "exec": ("Dynamic Code Execution", "Avoid exec() as it exposes arbitrary code execution risks."),
    "pickle.loads": ("Unsafe Pickle Deserialization", "Use JSON or safe serialization instead of pickle on untrusted data."),
}


class ASTSecurityVisitor(ast.NodeVisitor):
    """AST Visitor to detect unsafe function calls and deserializations."""

    def __init__(self, rel_path: str):
        self.rel_path = rel_path
        self.findings: List[SecurityFinding] = []

    def visit_Call(self, node: ast.Call):
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                func_name = f"{node.func.value.id}.{node.func.attr}"

        if func_name in UNSAFE_FUNCTION_NAMES:
            cat, rec = UNSAFE_FUNCTION_NAMES[func_name]
            self.findings.append(
                SecurityFinding(
                    file_path=self.rel_path,
                    line_number=node.lineno,
                    category=cat,
                    masked_value=f"Call to {func_name}()",
                    recommendation=rec,
                    severity="critical" if func_name in ("eval", "exec") else "high",
                    confidence="unambiguous",
                )
            )

        # Check for subprocess.Popen / run with shell=True
        if func_name in ("subprocess.Popen", "subprocess.run", "subprocess.call"):
            for keyword in node.keywords:
                if keyword.arg == "shell" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                    self.findings.append(
                        SecurityFinding(
                            file_path=self.rel_path,
                            line_number=node.lineno,
                            category="Shell Injection Risk",
                            masked_value="shell=True in subprocess call",
                            recommendation="Avoid shell=True; pass command arguments as a list of strings.",
                            severity="high",
                            confidence="likely",
                        )
                    )

        self.generic_visit(node)


def scan_file_for_secrets(file_path: str, rel_path: str) -> List[SecurityFinding]:
    findings: List[SecurityFinding] = []
    seen_matches = set()
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        for idx, line in enumerate(lines, 1):
            # Skip test mock files or harmless comments if needed
            for pattern, cat in SECRET_PATTERNS:
                match = re.search(pattern, line)
                if match:
                    # Extract the secret group or full match
                    secret_val = match.group(match.lastindex) if match.lastindex and match.lastindex >= 2 else match.group(0)
                    match_key = (idx, secret_val)
                    if match_key in seen_matches:
                        continue
                    seen_matches.add(match_key)

                    masked = mask_secret(secret_val)
                    findings.append(
                        SecurityFinding(
                            file_path=rel_path,
                            line_number=idx,
                            category=cat,
                            masked_value=masked,
                            recommendation="Move sensitive credentials to environment variables or a secrets manager.",
                            severity="critical",
                            confidence="likely",
                        )
                    )
    except Exception:
        pass
    return findings


def analyze(repository_id: str) -> List[SecurityFinding]:
    """
    Performs security static analysis scanning for hardcoded secrets and dangerous code execution.
    All detected secret values are strictly masked before returning.
    """
    repo_path = repository_id if os.path.exists(repository_id) else os.getcwd()
    all_findings: List[SecurityFinding] = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("venv", "node_modules")]
        for file in files:
            fpath = os.path.join(root, file)
            rel_path = os.path.relpath(fpath, repo_path)

            # 1. Regex scanning for secrets in code and configuration files
            if file.endswith((".py", ".js", ".ts", ".json", ".env", ".yaml", ".yml", ".toml", ".txt", ".md")):
                all_findings.extend(scan_file_for_secrets(fpath, rel_path))

            # 2. AST security scanning for Python files
            if file.endswith(".py"):
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    tree = ast.parse(content, filename=fpath)
                    visitor = ASTSecurityVisitor(rel_path)
                    visitor.visit(tree)
                    all_findings.extend(visitor.findings)
                except Exception:
                    continue

    return all_findings
