"""Lightweight OpenAI-compatible client using stdlib urllib.

Used for judge model calls and z.ai / OpenAI-compatible test subjects.
No external HTTP library dependencies.
"""

from __future__ import annotations

import json
import ssl
import time
from dataclasses import dataclass
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from tgb.clients.ollama_client import TimingData


@dataclass(frozen=True)
class OpenAIMessage:
    """A chat message for OpenAI-compatible APIs."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class OpenAICompatClient:
    """Lightweight OpenAI-compatible chat client."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        model: str = "",
        timeout: float = 180.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    def chat_complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.8,
        max_tokens: int = 2048,
        json_mode: bool = False,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion request, return the raw response dict."""
        payload: dict[str, Any] = {
            "model": model or self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        url = f"{self._base_url}/v1/chat/completions"
        return self._post_json(url, payload)

    def complete_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> tuple[str, TimingData]:
        """Higher-level: send system+user prompts, return (text, timing)."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        wall_start = time.monotonic()
        resp = self.chat_complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
        wall_elapsed = time.monotonic() - wall_start

        # Extract text from response
        text = ""
        choices = resp.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            text = message.get("content", "")

        # Extract usage
        usage = resp.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        eval_tps = 0.0
        if completion_tokens and wall_elapsed > 0:
            eval_tps = completion_tokens / wall_elapsed

        timing = TimingData(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            eval_tokens_per_sec=round(eval_tps, 1),
            wall_clock_seconds=round(wall_elapsed, 2),
        )

        return text, timing

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST JSON to URL and return parsed response."""
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        req = urllib_request.Request(url, data=data, headers=headers, method="POST")

        ctx = ssl.create_default_context()
        try:
            with urllib_request.urlopen(req, timeout=self._timeout, context=ctx) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib_error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI-compat API error {e.code}: {body}"
            ) from e
        except urllib_error.URLError as e:
            raise RuntimeError(
                f"OpenAI-compat connection error: {e.reason}"
            ) from e
