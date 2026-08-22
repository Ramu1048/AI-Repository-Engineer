"""
Analyzers package for AI Repository Engineer.
Exposes architecture, dependency, bug, and security static analysis tools.
"""

from analyzers.architecture import ArchitectureReport, analyze as analyze_architecture
from analyzers.bugs import BugFinding, analyze as analyze_bugs
from analyzers.dependencies import DependencyInfo, analyze as analyze_dependencies
from analyzers.security import SecurityFinding, analyze as analyze_security

__all__ = [
    "ArchitectureReport",
    "DependencyInfo",
    "BugFinding",
    "SecurityFinding",
    "analyze_architecture",
    "analyze_dependencies",
    "analyze_bugs",
    "analyze_security",
]
