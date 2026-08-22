"""
Architecture Analyzer Module.
Extracts modules, package import graphs, API boundaries, and generates Mermaid diagrams.
"""

import ast
from dataclasses import dataclass, field
import os
from typing import Dict, List, Optional, Set


@dataclass
class ArchitectureReport:
    modules: List[str] = field(default_factory=list)
    packages: List[str] = field(default_factory=list)
    dependency_graph: Dict[str, List[str]] = field(default_factory=dict)
    mermaid_diagram: str = ""
    api_boundaries: List[Dict[str, str]] = field(default_factory=list)
    summary: str = ""


def _parse_python_file(file_path: str, repo_root: str) -> tuple[str, List[str], List[Dict[str, str]]]:
    """Extracts module name, imported modules, and API boundaries from a Python file using AST."""
    rel_path = os.path.relpath(file_path, repo_root)
    module_name = rel_path.replace(os.sep, ".").rstrip(".py")

    imports: Set[str] = set()
    api_endpoints: List[Dict[str, str]] = []

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        tree = ast.parse(content, filename=file_path)

        for node in ast.walk(tree):
            # Imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])

            # Function / API boundary inspection
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_name = node.name
                # Check for API decorators like @app.get, @app.post, @router
                is_api = False
                decorator_str = ""
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                        dec_attr = dec.func.attr.lower()
                        if dec_attr in ("get", "post", "put", "delete", "patch", "route"):
                            is_api = True
                            decorator_str = f"@{dec.func.attr}"
                    elif isinstance(dec, ast.Name):
                        if "api" in dec.id.lower() or "endpoint" in dec.id.lower():
                            is_api = True
                            decorator_str = f"@{dec.id}"

                if is_api or func_name.startswith("api_") or func_name in ("ask", "analyze", "run"):
                    api_endpoints.append({
                        "file": rel_path,
                        "function": func_name,
                        "decorator": decorator_str or "public_function",
                    })

    except Exception:
        pass

    return module_name, list(imports), api_endpoints


def generate_mermaid_diagram(dependency_graph: Dict[str, List[str]]) -> str:
    """Generates a Mermaid graph TD string from a module dependency dictionary."""
    lines = ["graph TD"]
    edge_count = 0
    seen_edges = set()

    for source, targets in dependency_graph.items():
        clean_source = source.replace(".", "_").replace("-", "_")
        for target in targets:
            clean_target = target.replace(".", "_").replace("-", "_")
            if clean_source != clean_target and (clean_source, clean_target) not in seen_edges:
                seen_edges.add((clean_source, clean_target))
                lines.append(f"    {clean_source} --> {clean_target}")
                edge_count += 1
                if edge_count >= 30:  # Cap diagram size for readability
                    break
        if edge_count >= 30:
            break

    if len(lines) == 1:
        lines.append("    Root[Repository Root] --> Components[Modules]")

    return "\n".join(lines)


def analyze(repository_id: str) -> ArchitectureReport:
    """
    Analyzes the architectural hierarchy, dependencies, and API boundaries of a repository.
    """
    repo_path = repository_id if os.path.exists(repository_id) else os.getcwd()

    modules: List[str] = []
    packages: Set[str] = set()
    dependency_graph: Dict[str, List[str]] = {}
    all_api_boundaries: List[Dict[str, str]] = []

    for root, dirs, files in os.walk(repo_path):
        # Exclude hidden and venv directories
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("venv", "node_modules", "__pycache__")]
        rel_dir = os.path.relpath(root, repo_path)
        if rel_dir != ".":
            packages.add(rel_dir.replace(os.sep, "."))

        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                mod_name, imports, api_endpoints = _parse_python_file(full_path, repo_path)
                modules.append(mod_name)

                # Filter imports to internal modules
                internal_imports = [imp for imp in imports if imp in ("llm", "rag", "analyzers", "github_analysis", "retrieval", "github_service") or imp in mod_name]
                dependency_graph[mod_name] = internal_imports or imports[:5]
                all_api_boundaries.extend(api_endpoints)

    diagram = generate_mermaid_diagram(dependency_graph)
    summary = f"Parsed repository architecture: {len(modules)} modules across {len(packages)} package paths with {len(all_api_boundaries)} API entry points."

    return ArchitectureReport(
        modules=modules,
        packages=sorted(list(packages)),
        dependency_graph=dependency_graph,
        mermaid_diagram=diagram,
        api_boundaries=all_api_boundaries,
        summary=summary,
    )
