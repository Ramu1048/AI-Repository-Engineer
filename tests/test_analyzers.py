"""
Unit tests for Architecture, Dependency, Bug, and Security static analyzers.
"""

import os
import tempfile
import unittest

from analyzers.architecture import ArchitectureReport, analyze as analyze_architecture
from analyzers.bugs import BugFinding, analyze as analyze_bugs
from analyzers.dependencies import DependencyInfo, analyze as analyze_dependencies
from analyzers.security import SecurityFinding, analyze as analyze_security, mask_secret


class TestAnalyzers(unittest.TestCase):

    def test_secret_masking_function(self):
        self.assertEqual(mask_secret("sk-proj-1234567890abcdef"), "sk****ef")
        self.assertEqual(mask_secret("AKIAIOSFODNN7EXAMPLE"), "AK****LE")
        self.assertEqual(mask_secret("12345"), "1****5")
        self.assertEqual(mask_secret("abc"), "****")
        self.assertEqual(mask_secret(""), "")

    def test_security_analyzer_masks_secrets(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = os.path.join(tmp_dir, "config.py")
            raw_key = "sk-proj-9999888877776666"
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(f'API_KEY = "{raw_key}"\n')
                f.write("eval('import os')\n")

            findings = analyze_security(tmp_dir)

            self.assertGreaterEqual(len(findings), 1)
            for finding in findings:
                # Crucial requirement: RAW secret value must NEVER appear in finding outputs
                self.assertNotIn(raw_key, finding.masked_value)
                self.assertNotIn(raw_key, finding.recommendation)

            secret_findings = [f for f in findings if "Hardcoded" in f.category or "API Key" in f.category]
            self.assertEqual(len(secret_findings), 1)
            self.assertEqual(secret_findings[0].masked_value, "sk****66")

    def test_bug_analyzer_ast(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = os.path.join(tmp_dir, "bad_code.py")
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("def foo(items=[]):\n")
                f.write("    try:\n")
                f.write("        x = 10 / 0\n")
                f.write("    except:\n")
                f.write("        pass\n")

            findings = analyze_bugs(tmp_dir)
            self.assertGreaterEqual(len(findings), 3)

            codes = [f.issue_code for f in findings]
            self.assertIn("B006", codes)   # Mutable default
            self.assertIn("ZE001", codes)  # Division by zero
            self.assertIn("W0702", codes)  # Bare except

    def test_dependency_analyzer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            req_file = os.path.join(tmp_dir, "requirements.txt")
            with open(req_file, "w", encoding="utf-8") as f:
                f.write("requests>=2.28.0\npytest==7.1.2\n")

            pkg_file = os.path.join(tmp_dir, "package.json")
            with open(pkg_file, "w", encoding="utf-8") as f:
                f.write('{"dependencies": {"express": "^4.18.1"}}')

            deps = analyze_dependencies(tmp_dir)
            self.assertEqual(len(deps), 3)
            dep_names = [d.name for d in deps]
            self.assertIn("requests", dep_names)
            self.assertIn("pytest", dep_names)
            self.assertIn("express", dep_names)

    def test_dependency_analyzer_maven_ros(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pom_file = os.path.join(tmp_dir, "pom.xml")
            with open(pom_file, "w", encoding="utf-8") as f:
                f.write(
                    '<project><dependencies>'
                    '<dependency><groupId>org.springframework</groupId><artifactId>spring-web</artifactId><version>5.3.10</version></dependency>'
                    '</dependencies></project>'
                )

            ros_file = os.path.join(tmp_dir, "package.xml")
            with open(ros_file, "w", encoding="utf-8") as f:
                f.write('<package><depend>roscpp</depend><build_depend>std_msgs</build_depend></package>')

            deps = analyze_dependencies(tmp_dir)
            self.assertEqual(len(deps), 3)
            dep_names = [d.name for d in deps]
            self.assertIn("org.springframework:spring-web", dep_names)
            self.assertIn("roscpp", dep_names)
            self.assertIn("std_msgs", dep_names)

    def test_bug_analyzer_linter_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = os.path.join(tmp_dir, "test.py")
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("x = 1\n")

            # Must execute without throwing exceptions even if linters are missing or fail
            findings = analyze_bugs(tmp_dir)
            self.assertIsInstance(findings, list)

    def test_architecture_analyzer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mod_a = os.path.join(tmp_dir, "module_a.py")
            with open(mod_a, "w", encoding="utf-8") as f:
                f.write("import module_b\n\ndef ask(): pass\n")

            mod_b = os.path.join(tmp_dir, "module_b.py")
            with open(mod_b, "w", encoding="utf-8") as f:
                f.write("x = 1\n")

            report = analyze_architecture(tmp_dir)
            self.assertIsInstance(report, ArchitectureReport)
            self.assertIn("graph TD", report.mermaid_diagram)
            self.assertGreaterEqual(len(report.modules), 2)


if __name__ == "__main__":
    unittest.main()

