# selfcoder/embeddings.py
"""Embedding providers. The default speaks the OpenAI /v1/embeddings API."""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

from selfcoder.config import Config
from selfcoder.llm import LLMError


class Embedder(ABC):
    """A batch-capable embedder."""

    dim: int = 0

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class OpenAIEmbedder(Embedder):
    """Works with any server exposing the OpenAI /v1/embeddings schema."""

    def __init__(self, config: Config):
        self.config = config
        self.model = config.embedding_model
        self.batch_size = config.embedding_batch_size
        self.base_url = (config.embedding_base_url or config.base_url).rstrip("/")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
   
        key = self.config.api_key
        if not key and not self.config.api_key_optional:
            raise LLMError(
                f"No API key found for embeddings. Export ${self.config.api_key_env}, "
                f"or set embedding_provider = \"hashing\"."
            )

        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"

        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            body = {"model": self.model, "input": batch}
            request = urllib.request.Request(
                f"{self.base_url}/embeddings",
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:2000]
                if exc.code == 400 and "Pooling type 'none' is not OAI compatible" in detail:
                    raise LLMError(
                        f"Embedding server at {self.base_url} uses pooling 'none', "
                        "which is incompatible with /v1/embeddings. Restart the "
                        "embedding server with the pooling mode required by its "
                        "embedding model, or point SELFCODER_EMBEDDING_BASE_URL "
                        "at a compatible embedding server. For offline lexical "
                        "retrieval, set SELFCODER_EMBEDDING_PROVIDER=hashing and "
                        "run selfcoder index --force."
                    ) from exc
                raise LLMError(f"embedding HTTP {exc.code}: {detail}") from exc
            except urllib.error.URLError as exc:
                raise LLMError(f"embedding request failed: {exc.reason}") from exc
            except TimeoutError as exc:
                raise LLMError(f"embedding request timed out after {self.config.timeout}s.") from exc

            data = sorted(payload.get("data") or [], key=lambda d: d.get("index", 0))
            if len(data) != len(batch):
                raise LLMError(
                    f"embedding response had {len(data)} vectors for {len(batch)} inputs"
                )
            for item in data:
                vector = item.get("embedding") or []
                if not vector:
                    raise LLMError(f"embedding response missing a vector: {item!r}")
                out.append([float(x) for x in vector])

        if out:
            self.dim = len(out[0])
        return out


class HashingEmbedder(Embedder):
    """Offline, deterministic, pure-stdlib fallback.

    Token-hashing into a fixed-dim bag-of-words. Lexical only - it will match
    keywords but not paraphrase - but it needs no network and no API key, so
    the whole pipeline stays runnable in a fresh checkout.
    """

    _token_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]+|\d+")

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in self._token_re.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[index] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]


def build_embedder(config: Config) -> Embedder:
    """Choose an embedder, falling back to hashing if no API key is available."""
    if config.embedding_provider == "hashing":
        return HashingEmbedder(config.embedding_dim)
    if not config.api_key and not config.api_key_optional:
        # Don't crash; keep the pipeline usable and warn once.
        import sys

        print(
            "[selfcoder] No API key found - falling back to the offline "
            "hashing embedder (lexical retrieval only).",
            file=sys.stderr,
        )
        return HashingEmbedder(config.embedding_dim)
    return OpenAIEmbedder(config)
