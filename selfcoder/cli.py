# selfcoder/cli.py
"""Command line interface for the self-analysing / self-modifying agent."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from selfcoder import __version__
from selfcoder.agent import (
    analyze,
    ask,
    propose,
    remember_analysis,
    remember_edit,
    remember_lesson,
)
from selfcoder.codebase import iter_source_files, project_root, read_codebase
from selfcoder.config import Config
from selfcoder.embeddings import build_embedder
from selfcoder.llm import LLMClient, LLMError
from selfcoder.memory import MemoryStore, index_codebase
from selfcoder.patcher import Edit, PatchError, Patcher


# --------------------------------------------------------------------- colours

def _colour(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text


def bold(t: str) -> str:  return _colour("1", t)
def dim(t: str) -> str:   return _colour("2", t)
def green(t: str) -> str: return _colour("32", t)
def yellow(t: str) -> str:return _colour("33", t)
def red(t: str) -> str:   return _colour("31", t)
def cyan(t: str) -> str:  return _colour("36", t)


# --------------------------------------------------------------------- helpers

def open_store(config: Config, root: Path) -> MemoryStore:
    db_path = root / config.memory_dir / "memories.db"
    return MemoryStore(db_path, build_embedder(config))


def confirm(question: str) -> bool:
    if not sys.stdin.isatty():
        print(red("Refusing to modify files without a terminal to confirm. Use --yes."))
        return False
    try:
        answer = input(f"{question} [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in {"y", "yes"}


def print_report(report) -> None:
    for path in report.created:
        print(f"  {green('created')}  {path}")
    for path in report.changed:
        print(f"  {yellow('modified')} {path}")
    for path in report.deleted:
        print(f"  {red('deleted')}  {path}")
    if report.backup_dir:
        print(dim(f"  backup:  {report.backup_dir}"))


def print_diff(diff: str) -> None:
    if not diff:
        print(dim("(no changes)"))
        return
    for line in diff.splitlines():
        if line.startswith(("+++", "---")):
            print(bold(line))
        elif line.startswith("+"):
            print(green(line))
        elif line.startswith("-"):
            print(red(line))
        elif line.startswith("@@"):
            print(cyan(line))
        else:
            print(line)


def git_status(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root,
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def load_plan(path: Path) -> list[Edit]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read plan {path}: {exc}") from exc
    raw = data.get("edits") if isinstance(data, dict) else data
    if not isinstance(raw, list):
        raise SystemExit("Plan must be a list of edits or an object with an 'edits' list.")
    return [Edit.from_dict(item) for item in raw]


# --------------------------------------------------------------------- commands

def cmd_index(args, config: Config, root: Path) -> int:
    store = open_store(config, root)
    files = read_codebase(root, max_file_bytes=config.max_file_bytes)
    print(dim(f"Indexing {len(files)} file(s) into {store.path}"))
    updated, skipped, chunks = index_codebase(
        store, files, force=args.force, verbose=not args.quiet
    )
    print()
    print(bold("Indexed"))
    print(f"  files updated : {updated}")
    print(f"  files skipped : {skipped}")
    print(f"  chunks written: {chunks}")
    print(f"  total memories: {store.count()}")
    store.close()
    return 0


def cmd_recall(args, config: Config, root: Path) -> int:
    store = open_store(config, root)
    hits = store.search(
        args.query,
        k=args.k,
        kinds=args.kind or None,
        source_like=args.source,
    )
    if not hits:
        print(dim("(no matches)"))
        store.close()
        return 0
    for hit in hits:
        header = f"[{hit.kind}]"
        if hit.source:
            header += f" {hit.source}"
        if hit.label:
            header += f" :: {hit.label}"
        print(f"{bold(header)}  {dim(f'score={hit.score:.3f}')}")
        print(hit.text[: max(20, args.width)])
        if len(hit.text) > args.width:
            print(dim("  ..."))
        print()
    store.close()
    return 0


def cmd_memories(args, config: Config, root: Path) -> int:
    store = open_store(config, root)
    counts = store.kind_counts()
    print(bold("Memory store"))
    print(f"  path : {store.path}")
    print(f"  total: {store.count()}")
    if counts:
        print(bold("\nBy kind"))
        for kind, count in counts:
            print(f"  {kind:<14} {count}")

    if args.kind or args.source:
        print()
        print(bold("Recent"))
        for hit in store.list(kind=args.kind, source_like=args.source, limit=args.limit):
            tag = f"[{hit.kind}]"
            src = f" {hit.source}" if hit.source else ""
            lbl = f" :: {hit.label}" if hit.label else ""
            print(f"  {tag}{src}{lbl}")
    store.close()
    return 0


def cmd_forget(args, config: Config, root: Path) -> int:
    store = open_store(config, root)
    if not (args.id or args.kind or args.source):
        print(red("Refusing to delete without --id, --kind or --source."))
        return 2
    removed = store.delete(ids=args.id, kind=args.kind, source_like=args.source)
    print(f"{bold('Removed')} {removed} memory(ies).")
    store.close()
    return 0


def cmd_remember(args, config: Config, root: Path) -> int:
    store = open_store(config, root)
    memory_id = store.add(
        args.kind, args.text, source=args.source, label=args.label or "manual note"
    )
    if memory_id is None:
        print(dim("(duplicate - nothing stored)"))
    else:
        print(f"{bold('Stored')} memory #{memory_id}.")
    store.close()
    return 0


def cmd_analyze(args, config: Config, root: Path) -> int:
    client = LLMClient(config)
    store = open_store(config, root)

    if store.count() == 0:
        print(yellow("Memory store is empty. Run `selfcoder index` first for best results."))

    print(dim(f"Retrieving relevant excerpts and asking {config.model}..."))
    result = analyze(
        client, store, focus=args.focus,
        k=config.retrieval_k, budget=config.retrieval_budget_chars,
    )

    if args.save_memory:
        memory_id = remember_analysis(store, result, focus=args.focus)
        if memory_id is not None:
            print(dim(f"Stored analysis as memory #{memory_id}."))

    if args.json:
        print(json.dumps(result, indent=2))
        store.close()
        return 0

    print()
    print(bold("Overview"))
    print(result.get("overview", "(none)"))
    print()
    print(bold("Architecture"))
    print(result.get("architecture", "(none)"))

    strengths = result.get("strengths") or []
    if strengths:
        print()
        print(bold("Strengths"))
        for item in strengths:
            print(f"  + {item}")

    issues = result.get("issues") or []
    if issues:
        print()
        print(bold("Issues"))
        palette = {"high": red, "medium": yellow, "low": dim}
        for issue in issues:
            sev = str(issue.get("severity", "?")).lower()
            colour = palette.get(sev, dim)
            print(f"  {colour(f'[{sev}]')} {issue.get('file', '?')}")
            print(f"      problem: {issue.get('problem', '')}")
            print(f"      fix:     {issue.get('fix', '')}")

    steps = result.get("next_steps") or []
    if steps:
        print()
        print(bold("Next steps"))
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")

    store.close()
    return 0


def cmd_improve(args, config: Config, root: Path) -> int:
    coding = config.for_coding()
    client = LLMClient(coding)
    patcher = Patcher(root)
    store = open_store(config, root)

    if store.count() == 0:
        print(yellow("Memory store is empty. Run `selfcoder index` first for best results."))

    print(dim(f"Asking {coding.model} ({coding.base_url}) how to: {args.goal}"))
    plan, edits = propose(
        client, store, args.goal,
        k=config.retrieval_k, budget=config.retrieval_budget_chars,
        root=root, max_file_bytes=config.max_file_bytes,
    )

    print()
    print(bold("Summary"))
    print(plan.get("summary", "(none)"))
    print()
    print(bold("Reasoning"))
    print(plan.get("reasoning", "(none)"))

    if args.save_plan:
        payload = {
            "goal": args.goal,
            "summary": plan.get("summary"),
            "reasoning": plan.get("reasoning"),
            "edits": [e.__dict__ for e in edits],
        }
        Path(args.save_plan).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(dim(f"\nSaved plan to {args.save_plan}"))

    if not edits:
        print()
        print(yellow("The model proposed no edits."))
        if args.remember and not args.dry_run:
            store.add("note", f"Goal: {args.goal}\nNo edits proposed: {plan.get('reasoning', '')}",
                      source=args.goal, label="empty plan")
        store.close()
        return 0

    print()
    print(bold(f"Proposed diff ({len(edits)} edit(s))"))
    report = patcher.preview(edits, check_syntax=not args.allow_syntax_errors)
    print_diff(report.diff)

    if args.dry_run:
        print()
        print(dim("Dry run: nothing was written."))
        store.close()
        return 0

    print()
    if not args.yes and not confirm(bold("Apply these changes?")):
        print(dim("Aborted."))
        store.close()
        return 1

    report = patcher.apply(edits, allow_syntax_errors=args.allow_syntax_errors)
    print()
    print(bold("Applied"))
    print_report(report)

    if args.verify:
        print()
        print(dim(f"Running verification: {args.verify}"))
        verify = args.verify
        if isinstance(verify, str):
            verify = verify.split()
        result = subprocess.run(verify, cwd=root, shell=False)
        if result.returncode != 0:
            print(red(f"\nVerification failed (exit {result.returncode}). Rolling back."))
            if report.backup_dir:
                patcher.restore(report.backup_dir)
                print(dim("Rolled back."))
            if args.remember:
                store.add(
                    "lesson",
                    f"Attempt to satisfy goal {args.goal!r} failed verification "
                    f"({args.verify!r}) and was rolled back.",
                    source=args.goal,
                    label="rollback",
                )
            store.close()
            return 1
        print(green("Verification passed."))

    if args.remember:
        memory_id = remember_edit(store, args.goal, plan, report)
        if memory_id is not None:
            print(dim(f"Stored edit as memory #{memory_id}."))
        # Re-index touched files so knowledge stays in sync.
        files_to_reindex = read_codebase(
            root, max_file_bytes=config.max_file_bytes,
            only_paths=set(report.changed) | set(report.created),
        )
        index_codebase(store, files_to_reindex, verbose=False)

    if report.backup_dir:
        print()
        print(dim(f"Roll back with: selfcoder rollback {report.backup_dir.relative_to(root)}"))

    store.close()
    return 0


def cmd_apply(args, config: Config, root: Path) -> int:
    patcher = Patcher(root)
    edits = load_plan(Path(args.plan))
    report = patcher.preview(edits, check_syntax=not args.allow_syntax_errors)
    print(bold(f"Diff for {len(edits)} edit(s)"))
    print_diff(report.diff)

    if args.dry_run:
        print()
        print(dim("Dry run: nothing was written."))
        return 0

    print()
    if not args.yes and not confirm(bold("Apply these changes?")):
        print(dim("Aborted."))
        return 1

    report = patcher.apply(edits, allow_syntax_errors=args.allow_syntax_errors)
    print()
    print(bold("Applied"))
    print_report(report)

    if args.reindex:
        store = open_store(config, root)
        files = read_codebase(root, max_file_bytes=config.max_file_bytes)
        index_codebase(store, files, verbose=False)
        print(dim("Re-indexed touched files."))
        store.close()
    return 0


def cmd_rollback(args, config: Config, root: Path) -> int:
    patcher = Patcher(root)
    restored = patcher.restore(Path(args.backup))
    print(bold(f"Restored {len(restored)} file(s):"))
    for path in restored:
        print(f"  {path}")
    return 0


def cmd_status(args, config: Config, root: Path) -> int:
    files = list(iter_source_files(root))
    total = sum(p.stat().st_size for _, p in files if p.exists())

    print(bold("Project root "), root)
    print(bold("Chat model   "), config.model, dim(f"({config.base_url})"))
    coding = config.for_coding()
    print(bold("Coding model "), coding.model, dim(f"({coding.base_url})"))
    print(bold("Embeddings   "), config.embedding_provider, dim(f"({config.embedding_model})"))
    print(bold("API key      "),
          green("present") if config.api_key else red(f"missing (${config.api_key_env})"))
    print(bold("Source files "), len(files), dim(f"~{total / 1024:.1f} KiB"))

    try:
        store = open_store(config, root)
        print(bold("Memory store "), store.path, dim(f"({store.count()} memories)"))
        for kind, count in store.kind_counts():
            print(f"  {kind:<14} {count}")
        store.close()
    except Exception as exc:
        print(bold("Memory store "), red(f"unavailable: {exc}"))

    state = git_status(root)
    if state is not None:
        print()
        print(bold("Git status"))
        print("  " + (state.replace("\n", "\n  ") if state else green("clean")))
    return 0


def cmd_ask(args, config: Config, root: Path) -> int:
    client = LLMClient(config)
    store = open_store(config, root)
    answer = ask(
        client, store, args.question,
        k=config.retrieval_k, budget=config.retrieval_budget_chars,
    )
    print(answer)
    store.close()
    return 0


def cmd_config(args, config: Config, root: Path) -> int:
    print(json.dumps(config.redacted(), indent=2))
    return 0


# --------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="selfcoder",
        description="An LLM CLI that reads, reviews and rewrites its own source code "
                    "using a filesystem-backed vector memory.",
    )
    parser.add_argument("--version", action="version", version=f"selfcoder {__version__}")
    parser.add_argument("--root", help="project root (default: the package's parent directory)")
    parser.add_argument("--model", help="override the configured chat model")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("help", help="show this help message")

    p = sub.add_parser("index", help="chunk and embed the codebase into the memory store")
    p.add_argument("--force", action="store_true", help="re-embed even unchanged files")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("analyze", help="review the codebase and report issues")
    p.add_argument("--focus", help="steer the review")
    p.add_argument("--json", action="store_true")
    p.add_argument("--save-memory", action="store_true", default=True,
                   help="store the analysis as a memory (default)")
    p.add_argument("--no-save-memory", dest="save_memory", action="store_false")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("improve", help="ask the model to change its own code")
    p.add_argument("goal")
    p.add_argument("-y", "--yes", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-syntax-errors", action="store_true")
    p.add_argument("--save-plan", metavar="FILE")
    p.add_argument("--verify", metavar="CMD",
                   help="shell command to run after applying; rolls back on failure")
    p.add_argument("--remember", action="store_true", default=True)
    p.add_argument("--no-remember", dest="remember", action="store_false")
    p.set_defaults(func=cmd_improve)

    p = sub.add_parser("apply", help="apply an edit plan from a JSON file")
    p.add_argument("plan")
    p.add_argument("-y", "--yes", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-syntax-errors", action="store_true")
    p.add_argument("--reindex", action="store_true", help="re-index after applying")
    p.set_defaults(func=cmd_apply)

    p = sub.add_parser("rollback", help="undo a previous apply using its backup")
    p.add_argument("backup")
    p.set_defaults(func=cmd_rollback)

    p = sub.add_parser("status", help="show project root, files and memory stats")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("ask", help="ask a read-only question about the codebase")
    p.add_argument("question")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("recall", help="semantic search over memories")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=8)
    p.add_argument("--kind", action="append", choices=None,
                   help="restrict to a kind (can repeat)")
    p.add_argument("--source", help="SQL LIKE pattern on the source column")
    p.add_argument("--width", type=int, default=400, help="max characters per hit")
    p.set_defaults(func=cmd_recall)

    p = sub.add_parser("memories", help="list memory store contents")
    p.add_argument("--kind")
    p.add_argument("--source", help="SQL LIKE pattern on the source column")
    p.add_argument("--limit", type=int, default=25)
    p.set_defaults(func=cmd_memories)

    p = sub.add_parser("forget", help="delete memories")
    p.add_argument("--id", type=int, action="append")
    p.add_argument("--kind")
    p.add_argument("--source", help="SQL LIKE pattern")
    p.set_defaults(func=cmd_forget)

    p = sub.add_parser("remember", help="store a note in memory")
    p.add_argument("text")
    p.add_argument("--kind", default="note",
                   choices=["note", "lesson", "analysis", "edit", "conversation"])
    p.add_argument("--source")
    p.add_argument("--label")
    p.set_defaults(func=cmd_remember)

    p = sub.add_parser("config", help="print the effective configuration")
    p.set_defaults(func=cmd_config)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "help":
        parser.print_help()
        return 0

    config = Config.load()
    if getattr(args, "model", None):
        config.model = args.model

    root = Path(args.root).resolve() if args.root else project_root()
    if not root.is_dir():
        print(red(f"Not a directory: {root}"), file=sys.stderr)
        return 2

    try:
        return args.func(args, config, root)
    except LLMError as exc:
        print(red(f"LLM error: {exc}"), file=sys.stderr)
        return 1
    except PatchError as exc:
        print(red(f"Patch error: {exc}"), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print()
        print(dim("Interrupted."))
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
