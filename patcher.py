# selfcoder/patcher.py
"""Turning an LLM edit plan into safe, reversible filesystem changes."""

from __future__ import annotations

import ast
import difflib
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

IGNORED_PARTS = {
    ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "env",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
    ".selfcoder", "node_modules", ".idea", ".vscode",
}

VALID_ACTIONS = {"write", "patch", "delete"}


class PatchError(RuntimeError):
    """An edit plan could not be applied safely."""


@dataclass
class Edit:
    file: str
    action: str = "write"
    content: str | None = None
    find: str | None = None
    replace: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "Edit":
        if not isinstance(data, dict):
            raise PatchError(f"each edit must be an object, got {type(data).__name__}")
        if "file" not in data:
            raise PatchError(f"edit is missing a 'file' key: {data!r}")

        action = data.get("action", "write")
        if action not in VALID_ACTIONS:
            raise PatchError(f"unknown action {action!r} (expected one of {sorted(VALID_ACTIONS)})")

        edit = cls(
            file=str(data["file"]),
            action=action,
            content=data.get("content"),
            find=data.get("find"),
            replace=data.get("replace"),
        )

        if action == "write" and edit.content is None:
            raise PatchError(f"{edit.file}: a 'write' edit needs 'content'")
        if action == "patch" and (edit.find is None or edit.replace is None):
            raise PatchError(f"{edit.file}: a 'patch' edit needs both 'find' and 'replace'")
        return edit


@dataclass
class Report:
    created: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    diff: str = ""
    backup_dir: Path | None = None

    @property
    def empty(self) -> bool:
        return not (self.created or self.changed or self.deleted)


def _unified_diff(rel: str, old: str, new: str) -> str:
    if old == new:
        return ""
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
        )
    )


class Patcher:
    def __init__(self, root: Path, backup_root: Path | None = None):
        self.root = Path(root).resolve()
        self.backup_root = backup_root or (self.root / ".selfcoder" / "backups")

    # ------------------------------------------------------------------ paths

    def _resolve(self, rel: str) -> tuple[str, Path]:
        rel = rel.replace("\\", "/").lstrip("/")
        parts = Path(rel).parts
        if any(part in IGNORED_PARTS for part in parts):
            raise PatchError(f"refusing to touch ignored path: {rel}")
        if ".." in parts:
            raise PatchError(f"path traversal is not allowed: {rel}")

        path = (self.root / rel).resolve()
        if not path.is_relative_to(self.root):
            raise PatchError(f"path escapes the project root: {rel}")
        return path.relative_to(self.root).as_posix(), path

    # ------------------------------------------------------------------ planning

    def plan(self, edits: list[Edit]) -> dict[str, str | None]:
        """Compute the full new content of every touched file (None == delete)."""
        planned: dict[str, str | None] = {}

        for edit in edits:
            rel, path = self._resolve(edit.file)

            if edit.action == "delete":
                if not path.exists():
                    raise PatchError(f"{rel}: cannot delete a file that does not exist")
                planned[rel] = None
                continue

            if edit.action == "write":
                planned[rel] = edit.content or ""
                continue

            # patch
            if rel in planned:
                base = planned[rel]
            elif path.exists():
                base = path.read_text(encoding="utf-8")
            else:
                raise PatchError(f"{rel}: cannot patch a file that does not exist")

            if base is None:
                raise PatchError(f"{rel}: cannot patch a file that is being deleted")

            occurrences = base.count(edit.find)
            if occurrences == 0:
                raise PatchError(
                    f"{rel}: the 'find' snippet was not found. "
                    f"It must match the current file exactly.\n--- snippet ---\n{edit.find[:400]}"
                )
            if occurrences > 1:
                raise PatchError(
                    f"{rel}: the 'find' snippet matches {occurrences} times; make it more specific."
                )

            planned[rel] = base.replace(edit.find, edit.replace, 1)

        return planned

    # ------------------------------------------------------------------ preview

    def preview(self, edits: list[Edit]) -> Report:
        return self._build_report(self.plan(edits))

    def _build_report(self, planned: dict[str, str | None]) -> Report:
        report = Report()
        diffs: list[str] = []

        for rel, new in planned.items():
            path = self.root / rel
            existed = path.exists()
            old = path.read_text(encoding="utf-8") if existed else ""

            if new is None:
                report.deleted.append(rel)
                diffs.append(_unified_diff(rel, old, ""))
                continue

            if not existed:
                report.created.append(rel)
            elif old != new:
                report.changed.append(rel)

            if old != new:
                diffs.append(_unified_diff(rel, old, new))

        report.diff = "\n".join(d for d in diffs if d)
        return report

    # ------------------------------------------------------------------ applying

    def apply(
        self,
        edits: list[Edit],
        *,
        dry_run: bool = False,
        allow_syntax_errors: bool = False,
    ) -> Report:
        planned = self.plan(edits)
        report = self._build_report(planned)

        if report.empty or dry_run:
            return report

        if not allow_syntax_errors:
            self._check_syntax(planned)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup = self.backup_root / timestamp
        backup.mkdir(parents=True, exist_ok=True)

        originals: list[tuple[Path, str | None]] = []

        try:
            for rel, new in planned.items():
                path = self.root / rel
                if path.exists():
                    destination = backup / rel
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, destination)
                    originals.append((path, path.read_text(encoding="utf-8")))
                else:
                    originals.append((path, None))

                if new is None:
                    path.unlink(missing_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(new, encoding="utf-8")
        except Exception:
            self._rollback(originals)
            raise

        manifest = {
            "root": str(self.root),
            "created": report.created,
            "changed": report.changed,
            "deleted": report.deleted,
            "timestamp": timestamp,
        }
        (backup / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        report.backup_dir = backup
        return report

    def _check_syntax(self, planned: dict[str, str | None]) -> None:
        for rel, new in planned.items():
            if new is None or not rel.endswith(".py"):
                continue
            try:
                ast.parse(new, filename=rel)
            except SyntaxError as exc:
                raise PatchError(
                    f"{rel}: the proposed code has a syntax error on line {exc.lineno}: {exc.msg}. "
                    f"Nothing was written."
                ) from exc

    def _rollback(self, originals: list[tuple[Path, str | None]]) -> None:
        for path, old in originals:
            try:
                if old is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_text(old, encoding="utf-8")
            except OSError:
                pass

    # ------------------------------------------------------------------ restore

    def restore(self, backup_dir: Path) -> list[str]:
        """Undo a previously applied patch set using its backup manifest."""
        backup_dir = Path(backup_dir)
        manifest_path = backup_dir / "manifest.json"
        if not manifest_path.exists():
            raise PatchError(f"no manifest.json in {backup_dir}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        restored: list[str] = []

        for rel in manifest.get("created", []):
            (self.root / rel).unlink(missing_ok=True)
            restored.append(rel)

        for rel in manifest.get("changed", []) + manifest.get("deleted", []):
            source = backup_dir / rel
            if source.exists():
                destination = self.root / rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                restored.append(rel)

        return restored
