import ast
import re
import logging

logger = logging.getLogger("code_intelligence.parser")

from typing import Optional, List, Dict, Tuple, Any, Union

class ParsedSymbol:
    """Represents a code structure parsed from a file."""
    def __init__(
        self,
        symbol_type: str,
        symbol_name: str,
        class_name: Optional[str],
        start_line: int,
        end_line: int,
        docstring: Optional[str],
        language: str,
        extra: dict = None
    ):
        self.symbol_type = symbol_type
        self.symbol_name = symbol_name
        self.class_name = class_name
        self.start_line = start_line
        self.end_line = end_line
        self.docstring = docstring
        self.language = language
        self.extra = extra or {}

    def to_dict(self) -> dict:
        return {
            "symbol_type": self.symbol_type,
            "symbol_name": self.symbol_name,
            "class_name": self.class_name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "docstring": self.docstring,
            "language": self.language,
            **self.extra
        }


# ==========================================
# PYTHON PARSER (AST-based)
# ==========================================

class PythonVisitor(ast.NodeVisitor):
    def __init__(self, content: str):
        self.content = content
        self.lines = content.splitlines()
        self.symbols = []
        self.class_stack = []
        self.imports = []
        self.function_depth = 0

    def get_docstring(self, node) -> Optional[str]:
        return ast.get_docstring(node)

    def get_node_range(self, node) -> tuple[int, int]:
        start = node.lineno
        if hasattr(node, "decorator_list") and node.decorator_list:
            start = min(start, node.decorator_list[0].lineno)
        end = getattr(node, "end_lineno", node.lineno)
        return start, end

    def visit_Import(self, node):
        if len(self.class_stack) == 0 and self.function_depth == 0:
            names = [alias.name for alias in node.names]
            self.imports.append({
                "type": "import",
                "names": names,
                "start_line": node.lineno,
                "end_line": getattr(node, "end_lineno", node.lineno)
            })
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if len(self.class_stack) == 0 and self.function_depth == 0:
            module = node.module or ""
            names = [alias.name for alias in node.names]
            self.imports.append({
                "type": "import_from",
                "module": module,
                "names": names,
                "start_line": node.lineno,
                "end_line": getattr(node, "end_lineno", node.lineno)
            })
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        start_line, end_line = self.get_node_range(node)
        doc = self.get_docstring(node)
        
        self.symbols.append(ParsedSymbol(
            symbol_type="class",
            symbol_name=node.name,
            class_name=None,
            start_line=start_line,
            end_line=end_line,
            docstring=doc,
            language="python"
        ))
        
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node):
        self.function_depth += 1
        self.handle_function(node)
        self.function_depth -= 1

    def visit_AsyncFunctionDef(self, node):
        self.function_depth += 1
        self.handle_function(node)
        self.function_depth -= 1

    def handle_function(self, node):
        start_line, end_line = self.get_node_range(node)
        doc = self.get_docstring(node)
        
        is_method = len(self.class_stack) > 0
        parent_class = self.class_stack[-1] if is_method else None
        symbol_type = "method" if is_method else "function"
        
        self.symbols.append(ParsedSymbol(
            symbol_type=symbol_type,
            symbol_name=node.name,
            class_name=parent_class,
            start_line=start_line,
            end_line=end_line,
            docstring=doc,
            language="python"
        ))
        self.generic_visit(node)


def parse_python(content: str) -> list[ParsedSymbol]:
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        raise ValueError(f"Syntax error in Python file: {e}")
    
    visitor = PythonVisitor(content)
    visitor.visit(tree)
    
    if visitor.imports:
        start_line = min(imp["start_line"] for imp in visitor.imports)
        end_line = max(imp["end_line"] for imp in visitor.imports)
        lines = content.splitlines()
        import_lines = lines[start_line-1:end_line]
        import_content = "\n".join(import_lines)
        
        visitor.symbols.append(ParsedSymbol(
            symbol_type="imports",
            symbol_name="imports",
            class_name=None,
            start_line=start_line,
            end_line=end_line,
            docstring=None,
            language="python",
            extra={"content": import_content}
        ))
        
    return visitor.symbols


# ==========================================
# FALLBACK BRACE-MATCHING PARSER
# ==========================================

class FallbackParser:
    def __init__(self, content: str, language: str):
        self.content = content
        self.language = language.lower()
        self.lines = content.splitlines()
        self.symbols = []

    def parse(self) -> list[ParsedSymbol]:
        self.parse_imports()
        self.parse_blocks()
        return self.symbols

    def parse_imports(self):
        import_lines = []
        for idx, line in enumerate(self.lines):
            line_stripped = line.strip()
            if self.language in ("javascript", "typescript"):
                if (line_stripped.startswith("import ") or 
                    line_stripped.startswith("import{") or
                    line_stripped.startswith("export ") or
                    "require(" in line_stripped):
                    import_lines.append(idx + 1)
            elif self.language == "java":
                if line_stripped.startswith("import ") or line_stripped.startswith("package "):
                    import_lines.append(idx + 1)
            elif self.language == "go":
                if line_stripped.startswith("import ") or line_stripped.startswith("package "):
                    import_lines.append(idx + 1)
            elif self.language == "rust":
                if line_stripped.startswith("use ") or line_stripped.startswith("extern crate "):
                    import_lines.append(idx + 1)
            elif self.language in ("c", "cpp"):
                if line_stripped.startswith("#include") or line_stripped.startswith("#define"):
                    import_lines.append(idx + 1)
        
        if import_lines:
            start_line = min(import_lines)
            end_line = max(import_lines)
            self.symbols.append(ParsedSymbol(
                symbol_type="imports",
                symbol_name="imports",
                class_name=None,
                start_line=start_line,
                end_line=end_line,
                docstring=None,
                language=self.language,
                extra={"content": "\n".join(self.lines[start_line-1:end_line])}
            ))

    def parse_blocks(self):
        brace_pairs = []
        stack = []
        
        state = "NORMAL"
        idx = 0
        n = len(self.content)
        
        line_starts = [0]
        for char in self.content:
            if char == '\n':
                line_starts.append(idx + 1)
            idx += 1
        
        def get_line_num(char_idx):
            import bisect
            return bisect.bisect_right(line_starts, char_idx)

        idx = 0
        while idx < n:
            char = self.content[idx]
            
            if state == "NORMAL":
                if char == '/' and idx + 1 < n and self.content[idx+1] == '/':
                    state = "COMMENT_LINE"
                    idx += 1
                elif char == '/' and idx + 1 < n and self.content[idx+1] == '*':
                    state = "COMMENT_BLOCK"
                    idx += 1
                elif char == "'":
                    state = "STRING_SINGLE"
                elif char == '"':
                    state = "STRING_DOUBLE"
                elif char == '`' and self.language in ("javascript", "typescript"):
                    state = "STRING_BACKTICK"
                elif char == '{':
                    stack.append((idx, get_line_num(idx)))
                elif char == '}':
                    if stack:
                        start_char_idx, start_line = stack.pop()
                        end_line = get_line_num(idx)
                        brace_pairs.append((start_char_idx, idx, start_line, end_line))
            elif state == "COMMENT_LINE":
                if char == '\n':
                    state = "NORMAL"
            elif state == "COMMENT_BLOCK":
                if char == '*' and idx + 1 < n and self.content[idx+1] == '/':
                    state = "NORMAL"
                    idx += 1
            elif state == "STRING_SINGLE":
                if char == "'" and (idx == 0 or self.content[idx-1] != '\\'):
                    state = "NORMAL"
            elif state == "STRING_DOUBLE":
                if char == '"' and (idx == 0 or self.content[idx-1] != '\\'):
                    state = "NORMAL"
            elif state == "STRING_BACKTICK":
                if char == '`' and (idx == 0 or self.content[idx-1] != '\\'):
                    state = "NORMAL"
            
            idx += 1

        brace_pairs.sort(key=lambda x: x[0])
        blocks_info = []
        
        for start_char, end_char, start_line, end_line in brace_pairs:
            pre_start = max(0, start_char - 120)
            pre_text = self.content[pre_start:start_char]
            
            # Strip comments to prevent matching keywords inside comments
            pre_text = re.sub(r'//.*', '', pre_text)
            pre_text = re.sub(r'/\*.*?\*/', '', pre_text, flags=re.DOTALL)
            
            parts = re.split(r'[;}{]', pre_text)
            sig_text = parts[-1].strip()
            sig_text = re.sub(r'\s+', ' ', sig_text)
            
            is_class = False
            is_func = False
            symbol_name = ""
            
            class_keywords = r"\b(class|interface|enum|struct|trait|impl)\b"
            class_match = re.search(rf"{class_keywords}\s+(\w+)", sig_text)
            if class_match:
                is_class = True
                symbol_name = class_match.group(2)
            else:
                if self.language == "go":
                    go_method_match = re.search(r'\bfunc\s*\([^)]*\)\s*(\w+)\s*\(', sig_text)
                    if go_method_match:
                        is_func = True
                        symbol_name = go_method_match.group(1)
                    else:
                        go_func_match = re.search(r'\bfunc\s+(\w+)\s*\(', sig_text)
                        if go_func_match:
                            is_func = True
                            symbol_name = go_func_match.group(1)
                elif self.language == "rust":
                    rust_match = re.search(r'\bfn\s+(\w+)\s*\(', sig_text)
                    if rust_match:
                        is_func = True
                        symbol_name = rust_match.group(1)
                else:
                    func_match = re.search(r'\bfunction\s+(\w+)', sig_text)
                    if func_match:
                        is_func = True
                        symbol_name = func_match.group(1)
                    else:
                        arrow_match = re.search(r'\b(const|let|var)\s+(\w+)\s*=\s*[^=]*=>', sig_text)
                        if arrow_match:
                            is_func = True
                            symbol_name = arrow_match.group(2)
                        else:
                            method_match = re.search(r'\b(\w+)\s*\([^)]*\)\s*$', sig_text)
                            if method_match:
                                potential_name = method_match.group(1)
                                if potential_name not in ("if", "for", "while", "catch", "switch", "synchronized"):
                                    is_func = True
                                    symbol_name = potential_name
            
            if is_class or is_func:
                blocks_info.append({
                    "start_char": start_char,
                    "end_char": end_char,
                    "start_line": start_line,
                    "end_line": end_line,
                    "is_class": is_class,
                    "is_func": is_func,
                    "symbol_name": symbol_name,
                    "class_name": None
                })
        
        for b in blocks_info:
            if b["is_func"]:
                enclosing_class = None
                min_size = float('inf')
                for c in blocks_info:
                    if c["is_class"] and c["start_char"] < b["start_char"] and c["end_char"] > b["end_char"]:
                        size = c["end_char"] - c["start_char"]
                        if size < min_size:
                            min_size = size
                            enclosing_class = c
                if enclosing_class:
                    b["class_name"] = enclosing_class["symbol_name"]
        
        for b in blocks_info:
            symbol_type = "class" if b["is_class"] else ("method" if b["class_name"] else "function")
            docstring = self.extract_docstring_before(b["start_line"])
            
            self.symbols.append(ParsedSymbol(
                symbol_type=symbol_type,
                symbol_name=b["symbol_name"],
                class_name=b["class_name"],
                start_line=b["start_line"],
                end_line=b["end_line"],
                docstring=docstring,
                language=self.language
            ))

    def extract_docstring_before(self, start_line: int) -> Optional[str]:
        comment_lines = []
        idx = start_line - 2
        while idx >= 0:
            line = self.lines[idx].strip()
            if line.startswith("//"):
                comment_lines.append(line[2:].strip())
            elif line.endswith("*/"):
                j = idx
                block_lines = []
                while j >= 0:
                    j_line = self.lines[j].strip()
                    if "/*" in j_line:
                        parts = j_line.split("/*", 1)
                        if len(parts) > 1:
                            block_lines.append(parts[1].replace("*/", "").strip())
                        break
                    else:
                        block_lines.append(j_line.lstrip("*").strip())
                    j -= 1
                comment_lines.extend(reversed(block_lines))
                idx = j
            else:
                break
            idx -= 1
        
        if comment_lines:
            return "\n".join(reversed(comment_lines))
        return None


# ==========================================
# TREE-SITTER INTEGRATION
# ==========================================

class TreeSitterParser:
    def __init__(self, content: str, language: str):
        self.content = content.encode("utf8")
        self.language_name = language.lower()
        self.symbols = []

    def parse(self) -> list[ParsedSymbol]:
        try:
            from tree_sitter import Language, Parser
            
            if self.language_name == "javascript":
                import tree_sitter_javascript as tsjs
                lang = Language(tsjs.language())
            elif self.language_name == "typescript":
                import tree_sitter_typescript as tsts
                lang = Language(tsts.language_ts())
            elif self.language_name == "java":
                import tree_sitter_java as tsjava
                lang = Language(tsjava.language())
            else:
                raise ValueError(f"Tree-sitter grammar not loaded for: {self.language_name}")

            parser = Parser(lang)
            tree = parser.parse(self.content)
            self._traverse(tree.root_node)
            return self.symbols
        except Exception as e:
            logger.debug(f"Tree-sitter parsing failed, falling back. Error: {e}")
            fallback = FallbackParser(self.content.decode("utf8"), self.language_name)
            return fallback.parse()

    def _traverse(self, node, class_name=None):
        node_type = node.type
        
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        
        current_class = class_name
        
        if node_type in ("class_declaration", "class_expression"):
            name_node = node.child_by_field_name("name")
            name = name_node.text.decode("utf8") if name_node else "AnonymousClass"
            self.symbols.append(ParsedSymbol(
                symbol_type="class",
                symbol_name=name,
                class_name=None,
                start_line=start_line,
                end_line=end_line,
                docstring=self._get_ts_docstring(node),
                language=self.language_name
            ))
            current_class = name

        elif node_type in ("function_declaration", "generator_function_declaration", "method_definition"):
            name_node = node.child_by_field_name("name")
            name = name_node.text.decode("utf8") if name_node else "anonymous"
            symbol_type = "method" if class_name else "function"
            self.symbols.append(ParsedSymbol(
                symbol_type=symbol_type,
                symbol_name=name,
                class_name=class_name,
                start_line=start_line,
                end_line=end_line,
                docstring=self._get_ts_docstring(node),
                language=self.language_name
            ))
            
        elif node_type == "lexical_declaration":
            for i in range(node.named_child_count):
                child = node.named_child(i)
                if child.type == "variable_declarator":
                    value_node = child.child_by_field_name("value")
                    if value_node and value_node.type == "arrow_function":
                        name_node = child.child_by_field_name("name")
                        name = name_node.text.decode("utf8") if name_node else "anonymous"
                        self.symbols.append(ParsedSymbol(
                            symbol_type="function",
                            symbol_name=name,
                            class_name=class_name,
                            start_line=start_line,
                            end_line=end_line,
                            docstring=self._get_ts_docstring(node),
                            language=self.language_name
                        ))
        
        for i in range(node.named_child_count):
            self._traverse(node.named_child(i), current_class)

    def _get_ts_docstring(self, node) -> Optional[str]:
        prev = node.prev_sibling
        if prev and prev.type in ("comment", "comment_block"):
            return prev.text.decode("utf8").strip()
        return None


# ==========================================
# PUBLIC API
# ==========================================

def parse_code_file(content: str, language: str) -> list[ParsedSymbol]:
    """
    Parses code content of a given language into a list of ParsedSymbol objects.
    Falls back gracefully if language grammars or tree-sitter is missing.
    """
    lang = language.lower()
    
    if not content.strip():
        return []

    if lang == "python":
        return parse_python(content)
    
    try:
        import tree_sitter
        parser = TreeSitterParser(content, lang)
        return parser.parse()
    except ImportError:
        logger.debug("tree-sitter is not installed. Using FallbackParser.")
        fallback = FallbackParser(content, lang)
        return fallback.parse()
