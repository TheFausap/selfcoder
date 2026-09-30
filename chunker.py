# selfcoder/chunker.py
"""Split source files into retrieval-sized chunks with semantic boundaries."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    path: str          # relative POSIX path
    label: str         # human label, e.g. "function patcher.apply"
    text: str
    start_line: int
    end_line: int

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


PY_EXTS = {".py", ".pyi"}
DOC_EXTS = {".md", ".rst", ".txt"}


def chunk_file(rel_path: str, source: str, *, max_lines: int = 120) -> list[Chunk]:
    ext = Path(rel_path).suffix.lower()
    if ext in PY_EXTS:
        return _chunk_python(rel_path, source, max_lines)
    if ext in DOC_EXTS:
        return _chunk_markdown(rel_path, source)
    return _chunk_lines(rel_path, source, max_lines=max_lines, overlap=10)


def _chunk_python(rel_path: str, source: str, max_lines: int) -> list[Chunk]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _chunk_lines(rel_path, source, max_lines=max_lines, overlap=10)

    lines = source.splitlines(keepends=True)
    chunks: list[Chunk] = []

    # Module header: everything before the first top-level definition.
    first_def = next(
        (
            node.lineno - 1
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ),
        None,
    )
    header_end = first_def if first_def is not None else len(lines)
    header = "".join(lines[:header_end]).rstrip()
    if header:
        chunks.append(Chunk(rel_path, "module header", header, 1, header_end))

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        start = node.lineno - 1
        end = node.end_lineno or node.lineno
        body = "".join(lines[start:end]).rstrip()
        if not body:
            continue

        is_class = isinstance(node, ast.ClassDef)
        span = len(body.splitlines())

        # Large classes: keep a short declaration chunk, then each method.
        if is_class and span > max_lines:
            decl_end = min(start + 20, end)
            decl = "".join(lines[start:decl_end]).rstrip()
            chunks.append(
                Chunk(rel_path, f"class {node.name} (declaration)", decl, start + 1, decl_end)
            )
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    s = sub.lineno - 1
                    e = sub.end_lineno or sub.lineno
                    chunks.append(
                        Chunk(
                            rel_path,
                            f"method {node.name}.{sub.name}",
                            "".join(lines[s:e]).rstrip(),
                            s + 1,
                            e,
                        )
                    )
        else:
            kind = "class" if is_class else "function"
            chunks.append(Chunk(rel_path, f"{kind} {node.name}", body, start + 1, end))

    return chunks


def _chunk_markdown(rel_path: str, source: str) -> list[Chunk]:
    lines = source.splitlines(keepends=True)
    chunks: list[Chunk] = []
    start = 0
    title = "intro"

    for i, line in enumerate(lines):
        if not line.startswith("#"):
            continue
        if i > start:
            text = "".join(lines[start:i]).strip()
            if text:
                chunks.append(Chunk(rel_path, title, text, start + 1, i))
        start = i
        title = line.lstrip("#").strip() or "section"

    tail = "".join(lines[start:]).strip()
    if tail:
        chunks.append(Chunk(rel_path, title, tail, start + 1, len(lines)))
    return chunks


def _chunk_lines(rel_path: str, source: str, *, max_lines: int, overlap: int) -> list[Chunk]:
    lines = source.splitlines(keepends=True)
    step = max(1, max_lines - overlap)
    chunks: list[Chunk] = []
    for i in range(0, len(lines), step):
        window = lines[i : i + max_lines]
        text = "".join(window).rstrip()
        if text:
            chunks.append(
                Chunk(rel_path, f"lines {i + 1}-{i + len(window)}", text, i + 1, i + len(window))
            )
        if i + max_lines >= len(lines):
            break
    return chunks
