import pytest
import logging
from ingestion.chunker import chunk_repository
from ingestion.parser import parse_code_file

# Define the LoadedFile contract stub
class LoadedFile:
    def __init__(self, file_path: str, content: str, language: str, size_bytes: int = 0):
        self.file_path = file_path
        self.content = content
        self.language = language
        self.size_bytes = size_bytes


def test_python_ast_parsing_and_chunking():
    content = """import os
import sys

# Some global statement
GLOBAL_CONST = "hello"

@my_decorator
class Calculator:
    \"\"\"An awesome calculator class.\"\"\"
    
    def __init__(self):
        \"\"\"Constructor method.\"\"\"
        self.value = 0

    def add(self, a, b):
        return a + b

def standalone_func():
    \"\"\"Standalone function.\"\"\"
    return True
"""
    files = [LoadedFile("calc.py", content, "python")]
    chunks = chunk_repository("test_repo", files, max_chunk_chars=3000)
    
    # Assert imports chunk is present
    imports_chunk = [c for c in chunks if c["symbol_type"] == "imports"]
    assert len(imports_chunk) == 1
    assert "import os" in imports_chunk[0]["content"]
    assert "import sys" in imports_chunk[0]["content"]
    assert imports_chunk[0]["start_line"] == 1
    assert imports_chunk[0]["end_line"] == 2
    
    # Assert class chunk is present (since Calculator fits within budget)
    calc_class_chunk = [c for c in chunks if c["symbol_type"] == "class" and c["symbol_name"] == "Calculator"]
    assert len(calc_class_chunk) == 1
    assert "class Calculator" in calc_class_chunk[0]["content"]
    # Calculator line range starts at the decorator!
    assert calc_class_chunk[0]["start_line"] == 7
    
    # Assert standalone function is present
    standalone_chunk = [c for c in chunks if c["symbol_type"] == "function" and c["symbol_name"] == "standalone_func"]
    assert len(standalone_chunk) == 1
    assert "standalone_func" in standalone_chunk[0]["content"]
    
    # Assert global code/variables chunk is present
    global_chunk = [c for c in chunks if c["symbol_type"] == "global_code"]
    assert len(global_chunk) == 1
    assert 'GLOBAL_CONST = "hello"' in global_chunk[0]["content"]
    
    # Verify metadata fields are complete
    for chunk in chunks:
        assert chunk["repository_id"] == "test_repo"
        assert chunk["file_path"] == "calc.py"
        assert chunk["language"] == "python"
        assert chunk["start_line"] <= chunk["end_line"]
        assert chunk["content"].strip() != ""


def test_oversized_symbol_splitting():
    # Long function exceeding 100 characters limit (for testing purposes)
    lines = [
        "def extremely_long_function():",
        "    # Line 1 of body",
        "    x = 1",
        "    y = 2",
        "    # Line 4 of body",
        "    z = 3",
        "    # Line 6 of body",
        "    return x + y + z"
    ]
    content = "\n".join(lines)
    files = [LoadedFile("long.py", content, "python")]
    
    # Use max_chunk_chars=60 (small budget to force splitting)
    chunks = chunk_repository("test_repo", files, max_chunk_chars=60)
    
    func_chunks = [c for c in chunks if c["symbol_type"] == "function" and c["symbol_name"] == "extremely_long_function"]
    assert len(func_chunks) > 1
    
    # Check that each sub-chunk preserves original metadata
    for chunk in func_chunks:
        assert chunk["symbol_name"] == "extremely_long_function"
        assert chunk["symbol_type"] == "function"
        assert chunk["class_name"] is None
        assert chunk["start_line"] <= chunk["end_line"]
        # Verify content contains lines from original content
        assert "extremely_long_function" in chunk["content"] or "Line" in chunk["content"] or "return" in chunk["content"]


def test_markdown_section_chunking():
    content = """# Main Title
Welcome to the project.

## Installation
Run:
`pip install .`

## Usage
Import and run the entry point.
"""
    files = [LoadedFile("README.md", content, "markdown")]
    chunks = chunk_repository("test_repo", files)
    
    doc_sections = [c for c in chunks if c["symbol_type"] == "documentation_section"]
    assert len(doc_sections) == 3
    
    title_sec = [s for s in doc_sections if s["symbol_name"] == "Main Title"][0]
    assert "Welcome to the project." in title_sec["content"]
    
    install_sec = [s for s in doc_sections if s["symbol_name"] == "Installation"][0]
    assert "`pip install .`" in install_sec["content"]


def test_configuration_section_chunking():
    # JSON Config
    json_content = '{\n  "database": {\n    "host": "localhost"\n  },\n  "logging": {\n    "level": "info"\n  }\n}'
    json_file = LoadedFile("config.json", json_content, "json")
    
    # YAML Config
    yaml_content = "database:\n  host: localhost\nlogging:\n  level: info"
    yaml_file = LoadedFile("config.yaml", yaml_content, "yaml")
    
    # TOML Config
    toml_content = "[database]\nhost = 'localhost'\n[logging]\nlevel = 'info'"
    toml_file = LoadedFile("config.toml", toml_content, "toml")
    
    chunks = chunk_repository("test_repo", [json_file, yaml_file, toml_file])
    
    # Check JSON
    json_chunks = [c for c in chunks if c["file_path"] == "config.json"]
    assert len(json_chunks) == 2
    assert any(c["symbol_name"] == "database" for c in json_chunks)
    assert any(c["symbol_name"] == "logging" for c in json_chunks)
    
    # Check YAML
    yaml_chunks = [c for c in chunks if c["file_path"] == "config.yaml"]
    assert len(yaml_chunks) == 2
    assert any(c["symbol_name"] == "database" for c in yaml_chunks)
    assert any(c["symbol_name"] == "logging" for c in yaml_chunks)
    
    # Check TOML
    toml_chunks = [c for c in chunks if c["file_path"] == "config.toml"]
    assert len(toml_chunks) == 2
    assert any(c["symbol_name"] == "database" for c in toml_chunks)
    assert any(c["symbol_name"] == "logging" for c in toml_chunks)


def test_malformed_file_handling(caplog):
    # Syntax error in Python
    bad_python = "def broken_func("
    files = [
        LoadedFile("good.py", "def good(): return 1", "python"),
        LoadedFile("bad.py", bad_python, "python")
    ]
    
    # This should run without raising SyntaxError and return chunks for good.py only
    chunks = chunk_repository("test_repo", files)
    
    good_chunks = [c for c in chunks if c["file_path"] == "good.py"]
    bad_chunks = [c for c in chunks if c["file_path"] == "bad.py"]
    
    assert len(good_chunks) == 1
    assert len(bad_chunks) == 0


def test_fallback_language_parser():
    # JS code using braces matching fallback parser
    js_content = """// Class comment block
class User {
  constructor(name) {
    this.name = name;
  }
  
  // Method comment
  getName() {
    return this.name;
  }
}

// Global helper function
function helper() {
  return true;
}
"""
    files = [LoadedFile("app.js", js_content, "javascript")]
    chunks = chunk_repository("test_repo", files, max_chunk_chars=3000)
    
    # Verify User class chunk
    user_class = [c for c in chunks if c["symbol_type"] == "class" and c["symbol_name"] == "User"]
    assert len(user_class) == 1
    assert "class User" in user_class[0]["content"]
    assert "Class comment block" in user_class[0]["content"] or user_class[0]["start_line"] > 0
    
    # Verify helper function chunk
    helper_func = [c for c in chunks if c["symbol_type"] == "function" and c["symbol_name"] == "helper"]
    assert len(helper_func) == 1
    assert "helper()" in helper_func[0]["content"]


def test_python_nested_and_multiline_imports():
    content = """import os
from sys import (
    path,
    argv
)

def my_func():
    import json
    return True
"""
    files = [LoadedFile("app.py", content, "python")]
    chunks = chunk_repository("test_repo", files, max_chunk_chars=3000)
    
    # Assert top-level imports are collected and nested is ignored
    imports = [c for c in chunks if c["symbol_type"] == "imports"]
    assert len(imports) == 1
    assert imports[0]["start_line"] == 1
    # Multiline import ends at line 5
    assert imports[0]["end_line"] == 5
    assert "import os" in imports[0]["content"]
    assert "path" in imports[0]["content"]
    assert "import json" not in imports[0]["content"]

    # Function is correctly chunked and not swallowed
    func = [c for c in chunks if c["symbol_type"] == "function" and c["symbol_name"] == "my_func"]
    assert len(func) == 1
    assert func[0]["start_line"] == 7
    assert func[0]["end_line"] == 9


def test_json_nested_keys():
    content = """{
  "database": {
    "logging": "info"
  },
  "logging": {
    "level": "info"
  }
}"""
    files = [LoadedFile("config.json", content, "json")]
    chunks = chunk_repository("test_repo", files, max_chunk_chars=3000)
    
    db_chunks = [c for c in chunks if c["symbol_name"] == "database"]
    assert len(db_chunks) == 1
    assert db_chunks[0]["start_line"] == 2
    assert db_chunks[0]["end_line"] == 4
    
    log_chunks = [c for c in chunks if c["symbol_name"] == "logging"]
    assert len(log_chunks) == 1
    assert log_chunks[0]["start_line"] == 5
    assert log_chunks[0]["end_line"] == 8


def test_toml_table_comments_and_brackets():
    content = """[database] # database settings
host = "localhost"

[[servers]]
ip = "127.0.0.1"
"""
    files = [LoadedFile("config.toml", content, "toml")]
    chunks = chunk_repository("test_repo", files, max_chunk_chars=3000)
    
    db_chunks = [c for c in chunks if c["symbol_name"] == "database"]
    assert len(db_chunks) == 1
    assert db_chunks[0]["start_line"] == 1
    assert db_chunks[0]["end_line"] == 3
    
    srv_chunks = [c for c in chunks if c["symbol_name"] == "servers"]
    assert len(srv_chunks) == 1
    assert srv_chunks[0]["start_line"] == 4
    assert srv_chunks[0]["end_line"] == 5


def test_markdown_comments_in_code_blocks():
    content = """# Title
Here is some code:
```python
# This is a comment, not a heading
x = 10
```

## Section 2
Hello.
"""
    files = [LoadedFile("readme.md", content, "markdown")]
    chunks = chunk_repository("test_repo", files, max_chunk_chars=3000)
    
    # We should have exactly 2 documentation sections
    doc_sections = [c for c in chunks if c["symbol_type"] == "documentation_section"]
    assert len(doc_sections) == 2
    assert any(c["symbol_name"] == "Title" for c in doc_sections)
    assert any(c["symbol_name"] == "Section 2" for c in doc_sections)
    
    # Verify the code block comment is part of the Title section
    title_sec = [c for c in doc_sections if c["symbol_name"] == "Title"][0]
    assert "# This is a comment" in title_sec["content"]
