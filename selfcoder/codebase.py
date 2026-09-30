# selfcoder/codebase.py
"""Reading the project's own source tree into a prompt-friendly form."""

from __future__ import annotations

import os
from pathlib import Path

IGNORE_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "env",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".eggs",
    ".selfcoder", "node_modules", "dist", "build", ".idea", ".vscode",
}

INCLUDE_EXTS = {
    ".py", ".pyi", ".md", ".rst", ".txt", ".toml", ".cfg", ".ini",
    ".json", ".yaml", ".yml", ".sh",
}


def project_root() -> Path:
    """The directory containing the `selfcoder` package."""
    return Path(__file__).resolve().parent.parent


def iter_source_files(root: Path, include_exts=INCLUDE_EXTS, ignore_dirs=IGNORE_DIRS):
    """Yield (relative_posix_path, absolute_path), pruning ignored directories."""
    root = Path(root).resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in ignore_dirs)
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.suffix.lower() not in include_exts:
                continue
            yield path.relative_to(root).as_posix(), path


def read_codebase(
    root: Path,
    *,
    max_file_bytes: int = 60_000,
    include_exts=INCLUDE_EXTS,
    ignore_dirs=IGNORE_DIRS,
    only_paths: set[str] | None = None,
) -> dict[str, str]:
    """Return {relative_path: file_contents} for the whole project."""
    files: dict[str, str] = {}
    for rel, path in iter_source_files(root, include_exts, ignore_dirs):
        if only_paths is not None and rel not in only_paths:
            continue
        try:
            size = path.stat().st_size
            if size > max_file_bytes:
                files[rel] = f"[omitted: {size} bytes exceeds the {max_file_bytes} byte limit]"
                continue
            files[rel] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            files[rel] = f"[unreadable: {exc}]"
    return files


def render(files: dict[str, str], max_chars: int = 200_000) -> str:
    """Flatten the codebase into a single annotated blob for the prompt."""
    chunks: list[str] = []
    used = 0
    for rel in sorted(files):
        text = files[rel]
        block = f"### FILE: {rel}\n```\n{text}\n```\n"
        if used + len(block) > max_chars:
            chunks.append(f"### FILE: {rel}\n[omitted: context budget exhausted]\n")
            continue
        chunks.append(block)
        used += len(block)
    return "\n".join(chunks)
