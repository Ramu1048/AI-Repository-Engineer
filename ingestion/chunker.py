import logging
import re
import json
import yaml
import toml

from ingestion.parser import parse_code_file

logger = logging.getLogger("code_intelligence.chunker")

def split_oversized_content(content: str, start_line: int, max_chars: int = 3000, overlap_chars: int = 500) -> list[tuple[str, int, int]]:
    """
    Splits content into sub-chunks.
    Returns a list of tuples: (sub_content, sub_start_line, sub_end_line)
    """
    lines = content.splitlines()
    if len(content) <= max_chars or not lines:
        return [(content, start_line, start_line + len(lines) - 1 if lines else start_line)]

    sub_chunks = []
    current_chunk_lines = []
    current_chars = 0
    chunk_start_line = start_line
    
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        line_len = len(line) + 1  # +1 for newline
        
        if current_chars + line_len > max_chars and current_chunk_lines:
            sub_content = "\n".join(current_chunk_lines)
            sub_chunks.append((
                sub_content,
                chunk_start_line,
                chunk_start_line + len(current_chunk_lines) - 1
            ))
            
            # Slide window with overlap
            overlap_lines = []
            overlap_chars_count = 0
            for ol in reversed(current_chunk_lines):
                if overlap_chars_count + len(ol) + 1 > overlap_chars:
                    break
                overlap_lines.append(ol)
                overlap_chars_count += len(ol) + 1
            overlap_lines.reverse()
            
            current_chunk_lines = overlap_lines + [line]
            current_chars = sum(len(ol) + 1 for ol in overlap_lines) + line_len
            chunk_start_line = start_line + idx - len(overlap_lines)
        else:
            current_chunk_lines.append(line)
            current_chars += line_len
            
        idx += 1
        
    if current_chunk_lines:
        sub_content = "\n".join(current_chunk_lines)
        sub_chunks.append((
            sub_content,
            chunk_start_line,
            chunk_start_line + len(current_chunk_lines) - 1
        ))
        
    return sub_chunks


def chunk_config_file(content: str, language: str, max_chars: int) -> list[dict]:
    """
    Splits config files (JSON, YAML, TOML) by key/block.
    """
    lines = content.splitlines()
    chunks = []
    lang = language.lower()

    if lang == "json":
        try:
            data = json.loads(content)
        except Exception:
            return []
        
        if isinstance(data, dict):
            # Detect base indentation of the first top-level key
            indent_size = None
            for line in lines:
                match = re.match(r'^(\s*)"[^"]+"\s*:', line)
                if match:
                    indent_size = len(match.group(1))
                    break
            if indent_size is None:
                indent_size = 2
            
            for key, val in data.items():
                start_line = 1
                key_pattern = re.compile(rf'^\s{{{indent_size}}}"{re.escape(key)}"\s*:')
                for idx, line in enumerate(lines):
                    if key_pattern.match(line):
                        start_line = idx + 1
                        break
                
                next_key_line = len(lines)
                for idx in range(start_line, len(lines)):
                    line = lines[idx]
                    if re.match(rf'^\s{{{indent_size}}}"[^"]+"\s*:', line):
                        next_key_line = idx
                        break
                end_line = next_key_line
                
                chunks.append({
                    "symbol_type": "config_section",
                    "symbol_name": key,
                    "class_name": None,
                    "start_line": start_line,
                    "end_line": end_line,
                    "content": "\n".join(lines[start_line-1:end_line])
                })

    elif lang in ("yaml", "yml"):
        try:
            data = yaml.safe_load(content)
        except Exception:
            return []
        
        if isinstance(data, dict):
            # Detect base indentation of the first top-level key
            indent_size = None
            for line in lines:
                match = re.match(r'^(\s*)[A-Za-z0-9_-]+\s*:', line)
                if match:
                    indent_size = len(match.group(1))
                    break
            if indent_size is None:
                indent_size = 0
            
            for key, val in data.items():
                start_line = 1
                key_pattern = re.compile(rf'^\s{{{indent_size}}}{re.escape(key)}\s*:')
                for idx, line in enumerate(lines):
                    if key_pattern.match(line):
                        start_line = idx + 1
                        break
                
                next_key_line = len(lines)
                for idx in range(start_line, len(lines)):
                    line = lines[idx]
                    if re.match(rf'^\s{{{indent_size}}}[A-Za-z0-9_-]+\s*:', line):
                        next_key_line = idx
                        break
                end_line = next_key_line
                
                chunks.append({
                    "symbol_type": "config_section",
                    "symbol_name": key,
                    "class_name": None,
                    "start_line": start_line,
                    "end_line": end_line,
                    "content": "\n".join(lines[start_line-1:end_line])
                })

    elif lang == "toml":
        try:
            data = toml.loads(content)
        except Exception:
            return []
        
        if isinstance(data, dict):
            table_lines = []
            for idx, line in enumerate(lines):
                # Support comment-suffixed table definitions and clean bracket names
                match = re.match(r'^\s*\[+([^\]]+)\]+\s*(?:#.*)?$', line)
                if match:
                    table_lines.append((idx + 1, match.group(1).strip()))
                    
            if table_lines:
                for i, (line_num, table_name) in enumerate(table_lines):
                    start_line = line_num
                    end_line = table_lines[i+1][0] - 1 if i + 1 < len(table_lines) else len(lines)
                    chunks.append({
                        "symbol_type": "config_section",
                        "symbol_name": table_name,
                        "class_name": None,
                        "start_line": start_line,
                        "end_line": end_line,
                        "content": "\n".join(lines[start_line-1:end_line])
                    })
            else:
                for key in data.keys():
                    start_line = 1
                    key_pattern = re.compile(rf'^\s*{re.escape(key)}\s*=')
                    for idx, line in enumerate(lines):
                        if key_pattern.match(line.strip()):
                            start_line = idx + 1
                            break
                    
                    next_key_line = len(lines)
                    for idx in range(start_line, len(lines)):
                        line = lines[idx].strip()
                        if re.match(r'^[A-Za-z0-9_-]+\s*=', line):
                            next_key_line = idx
                            break
                    end_line = next_key_line
                    
                    chunks.append({
                        "symbol_type": "config_section",
                        "symbol_name": key,
                        "class_name": None,
                        "start_line": start_line,
                        "end_line": end_line,
                        "content": "\n".join(lines[start_line-1:end_line])
                    })

    return chunks


def chunk_markdown_file(content: str) -> list[dict]:
    """
    Splits markdown headings into sections.
    """
    lines = content.splitlines()
    chunks = []
    
    heading_pattern = re.compile(r'^(#+)\s+(.+)$')
    sections = []
    
    in_code_block = False
    for idx, line in enumerate(lines):
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            
        if not in_code_block:
            match = heading_pattern.match(line)
            if match:
                level = len(match.group(1))
                name = match.group(2).strip()
                sections.append((idx + 1, level, name))
            
    if not sections:
        return [{
            "symbol_type": "documentation_section",
            "symbol_name": "document_body",
            "class_name": None,
            "start_line": 1,
            "end_line": len(lines),
            "content": content
        }]
        
    first_section_line = sections[0][0]
    if first_section_line > 1:
        preamble_lines = lines[0:first_section_line-1]
        preamble_content = "\n".join(preamble_lines)
        if preamble_content.strip():
            chunks.append({
                "symbol_type": "documentation_section",
                "symbol_name": "preamble",
                "class_name": None,
                "start_line": 1,
                "end_line": first_section_line - 1,
                "content": preamble_content
            })
            
    for i, (start_line, level, name) in enumerate(sections):
        end_line = sections[i+1][0] - 1 if i + 1 < len(sections) else len(lines)
        section_lines = lines[start_line-1:end_line]
        section_content = "\n".join(section_lines)
        
        chunks.append({
            "symbol_type": "documentation_section",
            "symbol_name": name,
            "class_name": None,
            "start_line": start_line,
            "end_line": end_line,
            "content": section_content
        })
        
    return chunks


def chunk_text_file(content: str, max_chars: int) -> list[dict]:
    """
    Splits unstructured text files line by line into chunks.
    """
    sub_chunks = split_oversized_content(content, 1, max_chars)
    chunks = []
    for sub_c, start, end in sub_chunks:
        chunks.append({
            "symbol_type": "text_block",
            "symbol_name": "text_block",
            "class_name": None,
            "start_line": start,
            "end_line": end,
            "content": sub_c
        })
    return chunks


def chunk_code_file(content: str, language: str, max_chars: int) -> list[dict]:
    """
    Parses code structurally and chunks functions, classes, and globals.
    """
    try:
        symbols = parse_code_file(content, language)
    except ValueError as e:
        # A syntactically malformed file will raise ValueError, log and skip (return empty)
        logger.warning(f"Syntax error while parsing code file: {e}")
        return []
    except Exception as e:
        logger.warning(f"Unexpected parser failure: {e}")
        return []

    lines = content.splitlines()
    classes = [s for s in symbols if s.symbol_type == "class"]
    methods_and_funcs = [s for s in symbols if s.symbol_type in ("method", "function")]
    imports = [s for s in symbols if s.symbol_type == "imports"]
    
    chunks = []
    covered_ranges = []
    
    # Process imports first
    for imp in imports:
        imp_content = imp.extra.get("content") or "\n".join(lines[imp.start_line-1:imp.end_line])
        chunks.append({
            "symbol_type": "imports",
            "symbol_name": "imports",
            "class_name": None,
            "start_line": imp.start_line,
            "end_line": imp.end_line,
            "content": imp_content
        })
        covered_ranges.append((imp.start_line, imp.end_line))
        
    class_methods = {}
    for mf in methods_and_funcs:
        if mf.symbol_type == "method" and mf.class_name:
            class_methods.setdefault(mf.class_name, []).append(mf)
            
    chunked_method_keys = set()
    
    # Process classes
    for cls in classes:
        cls_content = "\n".join(lines[cls.start_line-1:cls.end_line])
        
        if len(cls_content) <= max_chars:
            chunks.append({
                "symbol_type": "class",
                "symbol_name": cls.symbol_name,
                "class_name": None,
                "start_line": cls.start_line,
                "end_line": cls.end_line,
                "content": cls_content
            })
            covered_ranges.append((cls.start_line, cls.end_line))
            
            methods_in_cls = class_methods.get(cls.symbol_name, [])
            for m in methods_in_cls:
                chunked_method_keys.add((m.symbol_name, m.start_line, m.end_line))
        else:
            methods_in_cls = class_methods.get(cls.symbol_name, [])
            header_end = cls.end_line
            if methods_in_cls:
                first_method_start = min(m.start_line for m in methods_in_cls)
                if first_method_start > cls.start_line:
                    header_end = first_method_start - 1
            
            if header_end >= cls.start_line:
                header_content = "\n".join(lines[cls.start_line-1:header_end])
                if header_content.strip():
                    sub_headers = split_oversized_content(header_content, cls.start_line, max_chars)
                    for sub_c, sub_start, sub_end in sub_headers:
                        chunks.append({
                            "symbol_type": "class",
                            "symbol_name": cls.symbol_name,
                            "class_name": None,
                            "start_line": sub_start,
                            "end_line": sub_end,
                            "content": sub_c
                        })
                    covered_ranges.append((cls.start_line, header_end))
                    
    # Process functions & methods
    for mf in methods_and_funcs:
        if (mf.symbol_name, mf.start_line, mf.end_line) in chunked_method_keys:
            continue
            
        mf_content = "\n".join(lines[mf.start_line-1:mf.end_line])
        sub_units = split_oversized_content(mf_content, mf.start_line, max_chars)
        for sub_c, sub_start, sub_end in sub_units:
            chunks.append({
                "symbol_type": mf.symbol_type,
                "symbol_name": mf.symbol_name,
                "class_name": mf.class_name,
                "start_line": sub_start,
                "end_line": sub_end,
                "content": sub_c
            })
        covered_ranges.append((mf.start_line, mf.end_line))

    # Fallback global scope code
    uncovered_lines = set(range(1, len(lines) + 1))
    for start, end in covered_ranges:
        for l in range(start, end + 1):
            uncovered_lines.discard(l)
            
    if uncovered_lines:
        sorted_uncovered = sorted(list(uncovered_lines))
        ranges = []
        if sorted_uncovered:
            curr_start = sorted_uncovered[0]
            curr_prev = sorted_uncovered[0]
            for l in sorted_uncovered[1:]:
                if l == curr_prev + 1:
                    curr_prev = l
                else:
                    ranges.append((curr_start, curr_prev))
                    curr_start = l
                    curr_prev = l
            ranges.append((curr_start, curr_prev))
            
            for start, end in ranges:
                range_content = "\n".join(lines[start-1:end])
                if range_content.strip():
                    sub_globals = split_oversized_content(range_content, start, max_chars)
                    for sub_c, sub_start, sub_end in sub_globals:
                        chunks.append({
                            "symbol_type": "global_code",
                            "symbol_name": "global_scope",
                            "class_name": None,
                            "start_line": sub_start,
                            "end_line": sub_end,
                            "content": sub_c
                        })
                        
    return chunks


# ==========================================
# PUBLIC API ENTRY POINT
# ==========================================

def chunk_repository(repository_id: str, files: list, max_chunk_chars: int = 3000) -> list[dict]:
    """
    Parses and chunks all files in the repository.
    Handles malformed files gracefully and preserves rich metadata.
    """
    all_chunks = []
    
    for f in files:
        file_path = f.file_path
        content = f.content
        language = f.language.lower() if f.language else ""
        
        # Determine chunking strategy based on language/type
        try:
            if not content.strip():
                # Skip empty files silently or log
                logger.info(f"Skipping empty file: {file_path}")
                continue
                
            if language in ("python", "javascript", "typescript", "java", "go", "rust", "c", "cpp"):
                file_chunks = chunk_code_file(content, language, max_chunk_chars)
            elif language in ("markdown", "md"):
                file_chunks = chunk_markdown_file(content)
            elif language in ("json", "yaml", "yml", "toml"):
                file_chunks = chunk_config_file(content, language, max_chunk_chars)
                if not file_chunks:  # fallback if parsing config fails
                    file_chunks = chunk_text_file(content, max_chunk_chars)
            else:
                # Text fallback for unknown languages
                file_chunks = chunk_text_file(content, max_chunk_chars)
                
            # Populate common repository and file level metadata
            for chunk in file_chunks:
                chunk["repository_id"] = repository_id
                chunk["file_path"] = file_path
                chunk["language"] = language
                all_chunks.append(chunk)
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
            # Never abort the whole repo's processing over one bad file
            continue
            
    return all_chunks
