# selfcoder/config.py
"""Configuration loading: JSON file at ~/.config/selfcoder/config.json + env overrides."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(
    os.environ.get("SELFCODER_CONFIG", Path.home() / ".config" / "selfcoder" / "config.json")
)

ENV_OVERRIDES = {
    "base_url": "SELFCODER_BASE_URL",
    "model": "SELFCODER_MODEL",
    "api_key_env": "SELFCODER_API_KEY_ENV",
    "temperature": "SELFCODER_TEMPERATURE",
    "max_tokens": "SELFCODER_MAX_TOKENS",
    "timeout": "SELFCODER_TIMEOUT",
    "embedding_model": "SELFCODER_EMBEDDING_MODEL",
    "embedding_provider": "SELFCODER_EMBEDDING_PROVIDER",
    "memory_dir": "SELFCODER_MEMORY_DIR",
    "retrieval_k": "SELFCODER_RETRIEVAL_K",
}


@dataclass
class Config:
    # chat model
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.2
    max_tokens: int = 8192
    timeout: int = 300

    # embeddings
    embedding_provider: str = "openai"   # "openai" | "hashing"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 512             # only used by the hashing embedder
    embedding_batch_size: int = 64
    embedding_base_url: str | None = None
    api_key_optional: bool = False

    # memory / retrieval
    memory_dir: str = ".selfcoder/memory"
    retrieval_k: int = 10
    retrieval_budget_chars: int = 60_000
    max_file_bytes: int = 60_000

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG_PATH) -> "Config":
        cfg = cls()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise SystemExit(f"Could not read config {path}: {exc}") from exc
            for key, value in data.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)

        for attr, env in ENV_OVERRIDES.items():
            raw = os.environ.get(env)
            if raw is None:
                continue
            current = getattr(cfg, attr)
            try:
                if isinstance(current, bool):
                    value = raw.lower() in {"1", "true", "yes", "on"}
                elif isinstance(current, int):
                    value = int(raw)
                elif isinstance(current, float):
                    value = float(raw)
                else:
                    value = raw
            except ValueError as exc:
                raise SystemExit(f"Bad value for {env}: {raw!r}") from exc
            setattr(cfg, attr, value)

        return cfg

    def save(self, path: Path = DEFAULT_CONFIG_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    def redacted(self) -> dict:
        data = asdict(self)
        data["api_key_present"] = bool(self.api_key)
        return data
