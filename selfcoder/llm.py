# selfcoder/llm.py
"""Minimal OpenAI-compatible chat client built on the standard library."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from selfcoder.config import Config


class LLMError(RuntimeError):
    """Any failure talking to the model."""


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from a model response."""
    text = (text or "").strip()
    if not text:
        raise LLMError("Model returned an empty response.")

    match = _FENCE_RE.search(text)
    if match:
        text = match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"Model did not return valid JSON: {exc}") from exc

    raise LLMError("Model did not return a JSON object.")


class LLMClient:
    def __init__(self, config: Config):
        self.config = config

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        key = self.config.api_key
        if not key and not self.config.api_key_optional:
            raise LLMError(
                f"No API key found. Export ${self.config.api_key_env}, "
                f"or set api_key_optional = true for a local server."
            )

        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:2000]
            raise LLMError(f"HTTP {exc.code} from {self.config.base_url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"Could not reach {self.config.base_url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise LLMError(f"Request timed out after {self.config.timeout}s.") from exc

        try:
            return body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected response shape: {str(body)[:500]}") from exc

    def chat_json(self, messages: list[dict], **kwargs) -> dict:
        return parse_json(self.chat(messages, json_mode=True, **kwargs))
