#!/usr/bin/env python3
import ast
import io
import os
import sys
from pathlib import Path
from typing import List, Tuple

IGNORED_DIRS = {".git", "__pycache__", ".venv", "venv", "build", "dist", "tools"}


def should_skip_dir(dirname: str) -> bool:
    return dirname in IGNORED_DIRS


def iter_python_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
        for fname in filenames:
            if fname.endswith(".py"):
                fpath = Path(dirpath) / fname
                # do not rewrite this script itself
                if fpath.name == "make_template.py" and fpath.parent.name == "tools":
                    continue
                yield fpath


def get_docstring_span(node: ast.AST) -> Tuple[int, int] | None:
    body = getattr(node, "body", [])
    if not body:
        return None
    first = body[0]
    if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) and isinstance(first.value.value, str):
        return first.lineno, getattr(first, "end_lineno", first.lineno)
    return None


def collect_function_nodes(tree: ast.AST) -> List[ast.AST]:
    funcs: List[ast.AST] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef):
            funcs.append(node)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            funcs.append(node)
            self.generic_visit(node)

    Visitor().visit(tree)
    # Process from bottom to top to keep line indices stable while editing
    funcs.sort(key=lambda n: n.lineno if hasattr(n, "lineno") else 0, reverse=True)
    return funcs


def leading_indent(s: str) -> str:
    i = 0
    while i < len(s) and s[i] in (" ", "\t"):
        i += 1
    return s[:i]


def find_body_indent(lines: List[str], start_line: int, end_line: int) -> str:
    # Search from the line after the signature to the end of node
    i = start_line  # 1-based incoming, will increment to next
    for ln in range(i, end_line + 1):
        text = lines[ln - 1]
        stripped = text.strip()
        if stripped == "":
            continue
        # Skip decorators or signature line occurrences by ensuring it's indented
        indent = leading_indent(text)
        if indent:
            return indent
    # Fallback to 4 spaces if cannot detect
    return "    "


def rewrite_file(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"[skip-syntax-error] {path}: {e}", file=sys.stderr)
        return False

    lines = src.splitlines(keepends=True)
    changed = False

    for fn in collect_function_nodes(tree):
        if not hasattr(fn, "lineno") or not hasattr(fn, "end_lineno"):
            continue
        def_line = fn.lineno
        end_line = fn.end_lineno  # inclusive

        # Determine body indent
        indent = find_body_indent(lines, def_line + 1, end_line)

        # Determine docstring span (if present)
        ds_span = get_docstring_span(fn)

        # Compute body start line to preserve docstring (or start at first body line)
        if ds_span is not None:
            ds_start, ds_end = ds_span
            preserve_from = ds_start
            preserve_to = ds_end
        else:
            preserve_from = None
            preserve_to = None

        # Find first non-empty, non-comment body line after signature to know where body starts
        # Minimal safe replacement range: from the line after preserved docstring (or first indented line) to end_line
        if preserve_to is not None:
            replace_start = preserve_to + 1
        else:
            # find the first indented (body) line after signature
            replace_start = def_line + 1
            while replace_start <= end_line and lines[replace_start - 1].strip() == "":
                replace_start += 1

        if replace_start > end_line:
            # Empty body (shouldn't happen), append a new line at end_line
            replace_start = end_line

        # Build new body content
        new_body_lines: List[str] = []
        if preserve_from is not None and preserve_to is not None:
            # Keep docstring lines as-is (they are before replace_start and will be preserved by not altering them)
            pass
        # Insert the NotImplementedError raise
        new_body_lines.append(f"{indent}raise NotImplementedError()\n")

        # Apply replacement: replace lines [replace_start, end_line] with new_body_lines
        # Convert to 0-based indices
        s_idx = replace_start - 1
        e_idx = end_line      # slice end is exclusive
        before = lines[:s_idx]
        after = lines[e_idx:]
        if lines[s_idx:e_idx] != new_body_lines:
            lines = before + new_body_lines + after
            changed = True

        # Update end positions of subsequent nodes implicitly handled by bottom-up order

    if changed:
        path.write_text("".join(lines), encoding="utf-8")
    return changed


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    total = 0
    changed = 0
    for py in iter_python_files(repo_root):
        total += 1
        if rewrite_file(py):
            changed += 1
    print(f"Transformed {changed} of {total} Python files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
