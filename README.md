# selfcoder

An LLM-powered CLI that reads, reviews and rewrites **its own source code**.

`selfcoder` treats its own project directory as the thing under analysis. It chunks
the codebase, embeds every chunk into a local vector database, and then asks a
language model to review, extend, or refactor itself. Each run stores what it
learned — analyses, edits, lessons — back into the same store, so the tool gets
more useful the more you use it.

It is deliberately conservative: every change is previewed as a diff, syntax-checked
before writing, backed up, and reversible with a single command.

---

## Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Configuration](#configuration)
  - [Using a local LLM (Ollama)](#using-a-local-llm-ollama)
  - [Using OpenAI or another hosted API](#using-openai-or-another-hosted-api)
  - [Environment variables](#environment-variables)
  - [Config file](#config-file)
- [Commands](#commands)
  - [index](#index)
  - [analyze](#analyze)
  - [improve](#improve)
  - [ask](#ask)
  - [apply](#apply)
  - [rollback](#rollback)
  - [status](#status)
  - [recall](#recall)
  - [memories](#memories)
  - [remember / forget](#remember--forget)
  - [config](#config)
- [How the memory works](#how-the-memory-works)
- [Safety model](#safety-model)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)
- [Extending it](#extending-it)

---

## Features

- **Retrieval-augmented, not context-stuffed.** The model sees the *relevant* chunks
  of code and prior notes, not the whole tree. Prompts stay small even as the
  project grows.
- **Filesystem-backed vector memory.** One SQLite file, no server, no external
  services, no extra Python dependencies.
- **Durable memories.** Analyses, edits, lessons and manual notes persist across
  runs and are retrieved by semantic similarity.
- **Code-aware chunking.** Python files are split on `ast` boundaries (module
  header, each class, each function, methods of large classes). Markdown splits
  on headings. Everything else falls back to overlapping line windows.
- **Incremental indexing.** Unchanged files are skipped via content hashes.
- **Safe by default.** Unified diff preview, `ast.parse` syntax gate, atomic
  apply, per-run backups, automatic rollback on verification failure.
- **Bring your own model.** Any server exposing the OpenAI `/v1/chat/completions`
  and `/v1/embeddings` schema works — Ollama, LM Studio, llama.cpp, vLLM,
  text-generation-webui, OpenAI itself.
- **Offline fallback.** With no API key, a pure-stdlib hashing embedder keeps
  the whole pipeline runnable (lexical retrieval instead of semantic).

---

## Requirements

- **Python 3.9 or newer.**
- **No third-party packages.** Standard library only — SQLite, `urllib`,
  `ast`, `argparse`, `difflib`.
- A **chat model endpoint** (local or hosted).
- Optionally, an **embedding model endpoint**. If none is available, `selfcoder`
  falls back to the offline hashing embedder automatically.

---

## Installation

Clone or download the project, then install it in editable mode from the
directory that contains `pyproject.toml`:

```bash
pip install -e .
```

This registers a `selfcoder` command on your `PATH`. You can also run it
without installing, as a module:

```bash
python -m selfcoder --help
```

---

## Quick start

Fully local, using Ollama:

```bash
# 1. Pull a chat model and an embedding model
ollama pull qwen3:8b
ollama pull qwen3-embedding:0.6b

# 2. Point selfcoder at your local server
export SELFCODER_BASE_URL=http://localhost:11434/v1
export SELFCODER_MODEL=qwen3:8b
export SELFCODER_EMBEDDING_MODEL=qwen3-embedding:0.6b
export SELFCODER_API_KEY_OPTIONAL=true

# 3. Build the vector index for the first time
selfcoder index

# 4. Review the codebase
selfcoder analyze --focus "error handling in the patcher"

# 5. Propose a change, preview the diff, and apply it after verification
selfcoder improve "add a --verbose flag that logs each retrieved chunk" \
    --verify "python -c 'import selfcoder'"
```

Against OpenAI:

```bash
export OPENAI_API_KEY=sk-...
selfcoder index
selfcoder analyze
selfcoder improve "cache embedding calls in memory.py" --dry-run
```

---

## Configuration

Configuration is resolved in this order (later wins):

1. Built-in defaults.
2. The config file at `~/.config/selfcoder/config.json` (override the location
   with `$SELFCODER_CONFIG`).
3. Environment variables.
4. Per-command flags such as `--model` and `--root`.

### Using a local LLM (Ollama)

Ollama serves both the chat and embeddings endpoints on the same port, so a
single `base_url` covers everything.

```bash
export SELFCODER_BASE_URL=http://localhost:11434/v1
export SELFCODER_MODEL=qwen3:8b
export SELFCODER_EMBEDDING_MODEL=qwen3-embedding:0.6b
export SELFCODER_API_KEY_OPTIONAL=true
```

Notes:

- **Use a dedicated embedding model.** A chat model such as Qwen3 is *not*
  suitable for embeddings. Qwen3-Embedding is a separate series
  (`0.6b`, `4b`, `8b`) trained specifically for retrieval. The `0.6b` model is
  small enough to run on CPU alongside your chat model and is more than
  adequate for a project this size.
- **No API key is needed.** Set `api_key_optional` (env:
  `SELFCODER_API_KEY_OPTIONAL=true`) so the client skips the `Authorization`
  header. Without it, `selfcoder` will refuse to start even though Ollama
  ignores the key.
- **If `/v1/embeddings` is unavailable** on your Ollama version, set
  `SELFCODER_EMBEDDING_PROVIDER=hashing` to use the offline lexical embedder.

Other local servers:

| Server | `SELFCODER_BASE_URL` | Notes |
|---|---|---|
| Ollama | `http://localhost:11434/v1` | Chat + embeddings |
| LM Studio | `http://localhost:1234/v1` | Enable the OpenAI-compatible server |
| llama.cpp | `http://localhost:8080/v1` | Pass `--api-key` unset on the server |
| vLLM | `http://localhost:8000/v1` | Chat + embeddings |
| text-generation-webui | `http://localhost:5000/v1` | Requires the OpenAI extension |

### Using OpenAI or another hosted API

```bash
export OPENAI_API_KEY=sk-...
# defaults are already:
#   SELFCODER_BASE_URL=https://api.openai.com/v1
#   SELFCODER_MODEL=gpt-4o-mini
#   SELFCODER_EMBEDDING_MODEL=text-embedding-3-small
```

You can also mix and match: chat against a local model, embeddings against a
hosted one. Set `embedding_base_url` in the config file (or
`SELFCODER_EMBEDDING_BASE_URL` in the environment) to send embedding requests
somewhere other than `base_url`:

```json
{
  "base_url": "http://localhost:11434/v1",
  "model": "qwen3:8b",
  "api_key_optional": true,
  "embedding_base_url": "https://api.openai.com/v1",
  "embedding_model": "text-embedding-3-small"
}
```

### Environment variables

| Variable | Maps to | Default |
|---|---|---|
| `SELFCODER_CONFIG` | config file path | `~/.config/selfcoder/config.json` |
| `SELFCODER_BASE_URL` | `base_url` | `https://api.openai.com/v1` |
| `SELFCODER_MODEL` | `model` | `gpt-4o-mini` |
| `SELFCODER_API_KEY_ENV` | name of the env var holding the key | `OPENAI_API_KEY` |
| `SELFCODER_API_KEY_OPTIONAL` | `api_key_optional` | `false` |
| `SELFCODER_TEMPERATURE` | `temperature` | `0.2` |
| `SELFCODER_MAX_TOKENS` | `max_tokens` | `8192` |
| `SELFCODER_TIMEOUT` | `timeout` (seconds) | `300` |
| `SELFCODER_EMBEDDING_PROVIDER` | `openai` or `hashing` | `openai` |
| `SELFCODER_EMBEDDING_MODEL` | `embedding_model` | `text-embedding-3-small` |
| `SELFCODER_EMBEDDING_BASE_URL` | `embedding_base_url` | falls back to `base_url` |
| `SELFCODER_MEMORY_DIR` | `memory_dir` (relative to project root) | `.selfcoder/memory` |
| `SELFCODER_RETRIEVAL_K` | `retrieval_k` | `10` |

### Config file

Create `~/.config/selfcoder/config.json`:

```json
{
  "base_url": "http://localhost:11434/v1",
  "model": "qwen3:8b",
  "api_key_optional": true,
  "embedding_provider": "openai",
  "embedding_model": "qwen3-embedding:0.6b",
  "embedding_dim": 512,
  "embedding_batch_size": 64,
  "memory_dir": ".selfcoder/memory",
  "retrieval_k": 10,
  "retrieval_budget_chars": 60000,
  "max_file_bytes": 60000,
  "temperature": 0.2,
  "max_tokens": 8192,
  "timeout": 300
}
```

`embedding_dim` is only used by the offline hashing embedder. Run
`selfcoder config` to print the effective configuration (the API key is shown
only as a boolean).

---

## Commands

### `index`

Chunk and embed the codebase into the memory store. Safe to run repeatedly:
files whose content hash hasn't changed are skipped.

```bash
selfcoder index            # incremental
selfcoder index --force    # re-embed everything
selfcoder index --quiet    # no per-file output
```

Run this after any change you made outside `selfcoder` (a pull, a manual edit).
`improve` re-indexes touched files for you automatically.

### `analyze`

Ask the model to review the code. Retrieval is keyed on architecture-level terms
plus your `--focus` string, and prior review notes are included so the model can
report what has and hasn't been fixed.

```bash
selfcoder analyze
selfcoder analyze --focus "concurrency and file locking"
selfcoder analyze --json > review.json
```

### `improve`

Ask the model to change its own code. It retrieves relevant code and history
associated with the same goal, proposes an edit plan, validates Python syntax,
shows you a diff, and — after confirmation — applies it atomically. Memories
from other improvement goals are excluded from the edit-planning prompt.

```bash
selfcoder improve "add a --verbose flag to the index command"
selfcoder improve "support .yaml config files" --dry-run
selfcoder improve "make backup paths portable" \
    --verify "pytest -q" \
    --save-plan plan.json
```

Flags:

| Flag | Meaning |
|---|---|
| `-y`, `--yes` | Skip the confirmation prompt |
| `--dry-run` | Show the diff and stop |
| `--allow-syntax-errors` | Skip the `ast.parse` gate (not recommended) |
| `--save-plan FILE` | Write the edit plan as JSON |
| `--verify CMD` | Run `CMD` after applying; roll back on non-zero exit |
| `--no-remember` | Don't record the edit as a memory |

### `ask`

Read-only Q&A about the codebase. Answers are grounded in retrieved chunks and
prior notes.

```bash
selfcoder ask "which module owns the backup timestamp logic?"
selfcoder ask "what happens if two edits touch the same file?"
```

### `apply`

Apply an edit plan produced earlier with `--save-plan`. Useful for review
workflows where a human inspects the JSON before it's written to disk.

```bash
selfcoder apply plan.json --dry-run
selfcoder apply plan.json --reindex
```

### `rollback`

Undo a previously applied patch set from its backup directory. The exact path is
printed after every successful `improve`.

```bash
selfcoder rollback .selfcoder/backups/20250101-120000
```

### `status`

Show the project root, source file count, memory store size, per-kind memory
counts, and `git status` if the directory is a repository.

```bash
selfcoder status
```

### `recall`

Semantic search over every memory.

```bash
selfcoder recall "how does syntax checking work"
selfcoder recall "rollback" --kind lesson
selfcoder recall "retrieval budget" --kind code -k 5
selfcoder recall "style" --source "style%" --width 800
```

### `memories`

List what's stored. With no filters, shows per-kind counts.

```bash
selfcoder memories
selfcoder memories --kind analysis --limit 10
selfcoder memories --source "add a --verbose%"
```

### `remember` / `forget`

Add a note to memory, or delete existing memories.

```bash
selfcoder remember "prefer pathlib over os.path in new code" \
    --kind lesson --source style

selfcoder forget --kind analysis --source "focus:%"
selfcoder forget --id 42 --id 43
```

Kinds: `code`, `analysis`, `edit`, `lesson`, `note`, `conversation`.

`forget` refuses to run without a filter — it will never wipe the store by
accident.

### `config`

Print the effective configuration with the API key redacted.

```bash
selfcoder config
```

---

## How the memory works

Everything lives in a single SQLite file at
`<project root>/.selfcoder/memory/memories.db`. There is one table, `memories`,
with a packed `array('f')` embedding per row (a 1536-dim vector is about 6 KB).
Cosine similarity is computed in pure Python over the candidate rows, which is
comfortably fast for tens of thousands of chunks.

Two conceptual collections share the table, distinguished by `kind`:

- **Knowledge** — `kind = "code"`. Chunks of the current source tree. Refreshed
  by `index`; deduplicated by `(kind, source, content_hash)` and by a per-file
  hash stored in a `meta` table, so unchanged files cost nothing.
- **Memories** — `kind` in `analysis`, `edit`, `lesson`, `note`,
  `conversation`. Notes the agent writes to itself:
  - after `analyze`, a compact review record is stored;
  - after a successful `improve`, the goal, summary, reasoning, touched files and
    backup path are stored;
  - after a failed verification and rollback, a `lesson` records what went wrong.

Prompts retrieve code and historical records separately. Improvement planning
uses current source files selected by named functions and code retrieval, and
only includes `edit`, `lesson`, or `note` history whose source matches the current
goal (ignoring case and repeated whitespace). `analyze` and `ask` continue to use
broader historical retrieval. `retrieval_budget_chars` caps how much of each
goes into the prompt. Retrieved historical items are labelled
`[kind] source :: label (score …)` so the model can tell code from commentary.

Deduplication means it is always safe to re-run `index`, and it also means the
same lesson won't be stored twice.

The whole store is a regular file: copy it, back it up, check it into a private
repository, or delete it and re-index.

---

## Safety model

`selfcoder` assumes the model will sometimes be wrong.

1. **Preview.** Every `improve` shows a unified diff and waits for confirmation
   before touching disk. `--dry-run` stops there. Non-interactive shells are
   refused unless `--yes` is passed.
2. **Syntax gate.** Proposed `.py` files are parsed with `ast.parse` before any
   write. A syntax error aborts the entire batch.
3. **Atomic apply.** All edits are computed first (so a bad `find` string is
   caught before anything is written), then applied in one pass. Any write
   failure triggers a full rollback.
4. **Backups.** Each applied patch set is copied to
   `.selfcoder/backups/<timestamp>/` with a `manifest.json`, and
   `selfcoder rollback <dir>` restores it.
5. **Verification hook.** `--verify "pytest -q"` runs a command after applying.
   A non-zero exit rolls back automatically and records a `lesson`.
6. **Scope guard.** Paths containing `..` or resolving outside the project root
   are rejected. Writes into `.git`, `.venv`, `__pycache__`, `node_modules`
   and `.selfcoder` are refused.

Because of the syntax gate, avoid `--allow-syntax-errors` unless you are
deliberately working on a file that isn't valid Python yet.

---

## Project layout

```
selfcoder/
  __init__.py        version
  __main__.py        enables `python -m selfcoder`
  config.py          dataclass config, env + JSON loading
  llm.py             OpenAI-compatible chat client, JSON extraction
  embeddings.py      OpenAIEmbedder + offline HashingEmbedder
  chunker.py         ast/markdown/line chunking
  memory.py          SQLite vector store, indexing, search
  patcher.py         edit planning, preview, apply, rollback
  agent.py           prompts and retrieval-augmented operations
  cli.py             argparse wiring and command implementations
pyproject.toml       packaging and the `selfcoder` entry point
```

The project is intentionally small and dependency-free so that the model can
reason about it without loading more than a handful of chunks per prompt.

---

## Troubleshooting

**"No API key found."**
Either export the variable named by `api_key_env` (default `OPENAI_API_KEY`), or
set `api_key_optional = true` if the server doesn't check the key — as is the
case for Ollama, LM Studio and llama.cpp.

**Embeddings fail with an HTTP error but chat works.**
Your server may not expose `/v1/embeddings`. Either point
`embedding_base_url` at a server that does, or fall back to the offline
embedder: `export SELFCODER_EMBEDDING_PROVIDER=hashing`.

**Retrieval returns irrelevant chunks.**
Check that you are actually using a *dedicated* embedding model. Chat models
produce poor embeddings. For Qwen, pull `qwen3-embedding:0.6b` or larger and set
`embedding_model` accordingly. After changing embedding models, re-index with
`selfcoder index --force`.

**"The 'find' snippet was not found."**
The model's `find` string didn't match the file byte-for-byte. This usually
means the index is stale — run `selfcoder index`, then try again. If it persists,
check that the file hasn't been edited by hand since the last index.

**"The proposed code has a syntax error."**
The batch was rejected before any write. Re-run `improve`, ideally with a
smaller goal. If you are intentionally working on incomplete code, pass
`--allow-syntax-errors`.

**Everything is slow.**
The cosine scan is linear. For projects with a few thousand chunks this is
fine; for tens of thousands, raise `retrieval_k` modestly and reduce
`embedding_batch_size` if you are rate-limited, or switch to the hashing
embedder while iterating locally.

**I want to start over.**
Delete `.selfcoder/memory/memories.db` and run `selfcoder index`. To keep the
backups, delete only the database.

---

## Extending it

The natural next steps, roughly in order of value:

- A cross-encoder reranker: retrieve top 50, rerank to top 8.
- Periodic summarisation of `analysis` and `edit` memories into a rolling set
  of `lesson` memories.
- Tie each memory to the git SHA it was written against, so `recall` can prefer
  memories that are still relevant to the current commit.
- A `--interactive` REPL mode where `ask`, `recall` and `improve` share one
  session.
- Replacing the linear cosine scan with `sqlite-vec` or FAISS when the store
  grows past a few tens of thousands of chunks.

Because the whole system is self-hosting, the most in-character way to build any
of these is to ask `selfcoder improve` to do it.
