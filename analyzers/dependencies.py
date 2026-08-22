"""
Dependency Analyzer Module.
Parses multi-ecosystem package manifests: requirements.txt, pyproject.toml, package.json, pom.xml, CMakeLists.txt, etc.
"""

from dataclasses import dataclass
import json
import os
import re
from typing import List, Optional


@dataclass
class DependencyInfo:
    name: str
    version: str
    ecosystem: str
    is_direct: bool
    file_source: str
    outdated_status: Optional[str] = None


def parse_requirements_txt(content: str, file_source: str) -> List[DependencyInfo]:
    deps = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        # Match pattern: package==version, package>=version, package~=version
        match = re.match(r"^([a-zA-Z0-9_\-\.]+)\s*([~=><!]=?\s*[\w\.\*]+)?", line)
        if match:
            pkg_name = match.group(1)
            version_str = match.group(2).strip() if match.group(2) else "any"
            deps.append(
                DependencyInfo(
                    name=pkg_name,
                    version=version_str,
                    ecosystem="Python (pip)",
                    is_direct=True,
                    file_source=file_source,
                )
            )
    return deps


def parse_pyproject_toml(content: str, file_source: str) -> List[DependencyInfo]:
    deps = []
    # Extract dependencies under [tool.poetry.dependencies] or dependencies = [...]
    in_deps = False
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("[") and "dependencies" in line.lower():
            in_deps = True
            continue
        elif line.startswith("[") and in_deps:
            in_deps = False

        if in_deps and "=" in line and not line.startswith("#"):
            parts = line.split("=", 1)
            name = parts[0].strip().strip('"').strip("'")
            ver = parts[1].strip().strip('"').strip("'")
            if name:
                deps.append(
                    DependencyInfo(
                        name=name,
                        version=ver,
                        ecosystem="Python (Poetry/PyProject)",
                        is_direct=True,
                        file_source=file_source,
                    )
                )

        # Standalone list items e.g. "requests>=2.25.0"
        match = re.search(r'"([a-zA-Z0-9_\-\.]+)\s*([~=><!]=?\s*[\w\.\*]+)?"', line)
        if match and not in_deps and "dependencies" in line.lower():
            deps.append(
                DependencyInfo(
                    name=match.group(1),
                    version=match.group(2) if match.group(2) else "any",
                    ecosystem="Python (PEP621)",
                    is_direct=True,
                    file_source=file_source,
                )
            )
    return deps


def parse_package_json(content: str, file_source: str) -> List[DependencyInfo]:
    deps = []
    try:
        data = json.loads(content)
        for dep_type, is_direct in [("dependencies", True), ("devDependencies", False)]:
            dep_dict = data.get(dep_type, {})
            if isinstance(dep_dict, dict):
                for name, ver in dep_dict.items():
                    deps.append(
                        DependencyInfo(
                            name=name,
                            version=str(ver),
                            ecosystem="npm / Node.js",
                            is_direct=is_direct,
                            file_source=file_source,
                        )
                    )
    except json.JSONDecodeError:
        pass
    return deps


def parse_cmake_lists(content: str, file_source: str) -> List[DependencyInfo]:
    deps = []
    # Match find_package(PackageName ...)
    matches = re.findall(r"find_package\(\s*([a-zA-Z0-9_\-]+)\s*([\d\.]+)?", content, re.IGNORECASE)
    for name, ver in matches:
        deps.append(
            DependencyInfo(
                name=name,
                version=ver if ver else "system",
                ecosystem="CMake / C++",
                is_direct=True,
                file_source=file_source,
            )
        )
    return deps


def parse_pom_xml(content: str, file_source: str) -> List[DependencyInfo]:
    deps = []
    try:
        import xml.etree.ElementTree as ET
        content_clean = re.sub(r'xmlns="[^"]+"', '', content, count=1)
        root = ET.fromstring(content_clean)
        for dep_node in root.findall(".//dependency"):
            group_id = dep_node.findtext("groupId", "").strip()
            artifact_id = dep_node.findtext("artifactId", "").strip()
            version = dep_node.findtext("version", "any").strip()
            if artifact_id:
                name = f"{group_id}:{artifact_id}" if group_id else artifact_id
                deps.append(
                    DependencyInfo(
                        name=name,
                        version=version,
                        ecosystem="Java (Maven)",
                        is_direct=True,
                        file_source=file_source,
                    )
                )
    except Exception:
        matches = re.findall(
            r"<dependency>\s*<groupId>([^<]+)</groupId>\s*<artifactId>([^<]+)</artifactId>(?:\s*<version>([^<]+)</version>)?",
            content,
            re.DOTALL,
        )
        for g_id, a_id, ver in matches:
            deps.append(
                DependencyInfo(
                    name=f"{g_id}:{a_id}",
                    version=ver if ver else "any",
                    ecosystem="Java (Maven)",
                    is_direct=True,
                    file_source=file_source,
                )
            )
    return deps


def parse_package_xml(content: str, file_source: str) -> List[DependencyInfo]:
    deps = []
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(content)
        dep_tags = ["depend", "build_depend", "exec_depend", "test_depend", "run_depend"]
        for tag in dep_tags:
            for node in root.findall(tag):
                dep_name = node.text.strip() if node.text else ""
                if dep_name:
                    deps.append(
                        DependencyInfo(
                            name=dep_name,
                            version="system",
                            ecosystem="ROS / C++",
                            is_direct=(tag != "test_depend"),
                            file_source=file_source,
                        )
                    )
    except Exception:
        matches = re.findall(r"<(?:depend|build_depend|exec_depend|test_depend|run_depend)>([^<]+)</", content)
        for dep_name in matches:
            dep_name = dep_name.strip()
            if dep_name:
                deps.append(
                    DependencyInfo(
                        name=dep_name,
                        version="system",
                        ecosystem="ROS / C++",
                        is_direct=True,
                        file_source=file_source,
                    )
                )
    return deps


def analyze(repository_id: str) -> List[DependencyInfo]:
    """
    Parses manifest files in repository_id and returns identified dependencies.
    """
    repo_path = repository_id if os.path.exists(repository_id) else os.getcwd()
    results: List[DependencyInfo] = []

    manifest_names = {
        "requirements.txt": parse_requirements_txt,
        "pyproject.toml": parse_pyproject_toml,
        "package.json": parse_package_json,
        "CMakeLists.txt": parse_cmake_lists,
        "pom.xml": parse_pom_xml,
        "package.xml": parse_package_xml,
    }

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("venv", "node_modules")]
        for file in files:
            parser_func = manifest_names.get(file)
            if parser_func:
                fpath = os.path.join(root, file)
                rel_path = os.path.relpath(fpath, repo_path)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    parsed = parser_func(content, rel_path)
                    results.extend(parsed)
                except Exception:
                    continue

    return results

