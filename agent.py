# selfcoder/agent.py
"""Retrieval-augmented operations. The prompt sees excerpts, not the whole tree."""

from __future__ import annotations

import json
import time
from pathlib import Path

from selfcoder.llm import LLMClient
from selfcoder.memory import Hit, MemoryStore, render_hits
from selfcoder.patcher import Edit, PatchError

EDIT_RULES = """\
You are maintaining a real, runnable Python project. You will be shown excerpts
from the current source tree and prior memories, then asked to change it.

Rules:
1. Only modify files that appear in the excerpts, or create new files inside the
   project directory. Paths are POSIX-style and relative to the project root.
2. Never touch .git/, .venv/, __pycache__/, .selfcoder/ or node_modules/.
3. Prefer "patch" edits (a small exact find/replace) over rewriting whole files.
   Use "write" only for new files or near-total rewrites.
4. For a "patch", the "find" string MUST appear in the current file exactly once,
   character for character, including indentation and blank lines.
5. Never leave placeholders, "...", elisions or TODO stubs. Every file you write
   must be complete and syntactically valid Python.
6. Standard library only unless the goal explicitly requires a dependency.
7. If the change is already present or impossible, return an empty "edits" list
   and explain why in "reasoning".

Respond with a single JSON object:

{
  "summary": "one line describing the change",
  "reasoning": "a short paragraph explaining approach and trade-offs",
  "edits": [
    {"file": "selfcoder/cli.py", "action": "patch",
     "find": "exact existing snippet", "replace": "replacement snippet"},
    {"file": "selfcoder/new_module.py", "action": "write", "content": "full contents"}
  ]
}
"""

ANALYZE_SYSTEM = """\
You are a meticulous senior Python engineer reviewing a small CLI project.
Be concrete: cite file names, function names and line ranges from the excerpts
you are given. Prioritise correctness and safety over style.

You may be shown notes from earlier reviews - do not simply repeat them. If
something you previously flagged is now fixed, say so.

Respond with a single JSON object:

{
  "overview": "what this project does, in two or three sentences",
  "architecture": "how the modules fit together",
  "strengths": ["..."],
  "issues": [
    {"severity": "high|medium|low", "file": "path.py",
     "problem": "what is wrong", "fix": "how to fix it"}
  ],
  "next_steps": ["the highest-value changes, in priority order"]
}
"""


# --------------------------------------------------------------- retrieval


def retrieve(
    store: MemoryStore,
    query: str,
    *,
    k: int,
    budget: int,
    kinds: list[str] | None = None,
) -> str:
    hits = store.search(query, k=k, kinds=kinds)
    if not hits:
        return "(no relevant memories found - run `selfcoder index` if this is a fresh checkout)"
    return render_hits(hits, max_chars=budget)


def _codebase_block(store: MemoryStore, query: str, k: int, budget: int) -> str:
    return retrieve(store, query, k=k, budget=budget, kinds=["code"])


def _memory_block(store: MemoryStore, query: str, k: int, budget: int) -> str:
    return retrieve(
        store, query, k=k, budget=budget,
        kinds=["analysis", "edit", "lesson", "note", "conversation"],
    )


# --------------------------------------------------------------- commands


def analyze(
    client: LLMClient,
    store: MemoryStore,
    *,
    focus: str | None,
    k: int,
    budget: int,
) -> dict:
    query = "overview of architecture, main modules, entry points, error handling"
    if focus:
        query = f"{query}. Focus: {focus}"

    code = _codebase_block(store, query, k, budget)
    past = _memory_block(store, "past reviews of this project", k, budget)

    user = (
        f"Relevant code excerpts:\n\n{code}\n\n"
        f"Prior review notes:\n\n{past}\n\n"
        f"Review this project and report your findings."
    )
    if focus:
        user += f"\nPay particular attention to: {focus}"

    return client.chat_json(
        [
            {"role": "system", "content": ANALYZE_SYSTEM},
            {"role": "user", "content": user},
        ]
    )


def propose(
    client: LLMClient,
    store: MemoryStore,
    goal: str,
    *,
    k: int,
    budget: int,
) -> tuple[dict, list[Edit]]:
    code = _codebase_block(store, goal, k, budget)
    past = _memory_block(store, goal, k, budget)

    plan = client.chat_json(
        [
            {"role": "system", "content": EDIT_RULES},
            {
                "role": "user",
                "content": (
                    f"Relevant code excerpts:\n\n{code}\n\n"
                    f"Relevant prior memories (analyses, edits, lessons):\n\n{past}\n\n"
                    f"Goal: {goal}\n\n"
                    f"Produce the edit plan that accomplishes this goal."
                ),
            },
        ]
    )

    raw = plan.get("edits") or []
    if not isinstance(raw, list):
        raise PatchError("The model returned an 'edits' value that is not a list.")
    return plan, [Edit.from_dict(item) for item in raw]


def ask(
    client: LLMClient,
    store: MemoryStore,
    question: str,
    *,
    k: int,
    budget: int,
) -> str:
    code = _codebase_block(store, question, k, budget)
    past = _memory_block(store, question, k, budget)
    return client.chat(
        [
            {
                "role": "system",
                "content": (
                    "You answer questions about a Python codebase. Cite file and "
                    "function names. Say when you are unsure. You are read-only: "
                    "never claim to have changed anything."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Relevant code excerpts:\n\n{code}\n\n"
                    f"Relevant prior memories:\n\n{past}\n\n"
                    f"Question: {question}"
                ),
            },
        ]
    )


# --------------------------------------------------------------- remembering


def remember_analysis(
    store: MemoryStore, result: dict, *, focus: str | None = None
) -> int | None:
    summary = result.get("overview") or "analysis"
    issues = result.get("issues") or []
    body_lines = [f"OVERVIEW: {summary}", f"ARCHITECTURE: {result.get('architecture', '')}"]
    for issue in issues:
        body_lines.append(
            f"- [{issue.get('severity', '?')}] {issue.get('file', '?')}: "
            f"{issue.get('problem', '')} -> {issue.get('fix', '')}"
        )
    body = "\n".join(body_lines)
    label = f"analysis {time.strftime('%Y-%m-%d %H:%M')}"
    if focus:
        label += f" (focus: {focus})"

    return store.add(
        "analysis",
        body,
        source=focus or "general",
        label=label,
        metadata={"focus": focus, "issue_count": len(issues)},
    )


def remember_edit(store: MemoryStore, goal: str, plan: dict, report) -> int | None:
    touched = report.created + report.changed + report.deleted
    body_lines = [
        f"GOAL: {goal}",
        f"SUMMARY: {plan.get('summary', '')}",
        f"REASONING: {plan.get('reasoning', '')}",
        f"FILES: {', '.join(touched) or '(none)'}",
    ]
    label = f"edit {time.strftime('%Y-%m-%d %H:%M')}"
    return store.add(
        "edit",
        "\n".join(body_lines),
        source=goal,
        label=label,
        metadata={
            "created": report.created,
            "changed": report.changed,
            "deleted": report.deleted,
            "backup": str(report.backup_dir) if report.backup_dir else None,
        },
    )


def remember_lesson(store: MemoryStore, text: str, *, source: str | None = None) -> int | None:
    return store.add("lesson", text, source=source, label="lesson")
