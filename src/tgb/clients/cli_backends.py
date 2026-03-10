"""Thin wrappers around text-game-engine's CLI backends for benchmark use.

Each client wraps the engine's async backend in a synchronous interface
matching the CompletionClient protocol: complete(system, user) -> (str, TimingData).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from text_game_engine.backends.base import ChatMessage, CompletionRequest

from tgb.clients.ollama_client import TimingData


def _run_async(coro):
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def _build_messages(system_prompt: str, user_prompt: str) -> list[ChatMessage]:
    """Build ChatMessage list from system + user prompts."""
    msgs = []
    if system_prompt:
        msgs.append(ChatMessage(role="system", content=system_prompt))
    msgs.append(ChatMessage(role="user", content=user_prompt))
    return msgs


def _extract_timing(usage: dict[str, int] | None, wall_seconds: float) -> TimingData:
    """Extract timing data from completion result usage dict."""
    if not usage:
        return TimingData(wall_clock_seconds=round(wall_seconds, 2))

    prompt_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
    eval_tps = 0.0
    if completion_tokens and wall_seconds > 0:
        eval_tps = completion_tokens / wall_seconds

    return TimingData(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        eval_tokens_per_sec=round(eval_tps, 1),
        wall_clock_seconds=round(wall_seconds, 2),
    )


class ClaudeCLIClient:
    """Benchmark client wrapping text-game-engine's ClaudeCLIBackend."""

    def __init__(self, model: str | None = None, **kwargs: Any) -> None:
        from text_game_engine.backends.claude_cli import ClaudeCLIBackend
        self._model = model
        self._backend = ClaudeCLIBackend(model=model, **kwargs)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        **opts: Any,
    ) -> tuple[str, TimingData]:
        request = CompletionRequest(
            messages=_build_messages(system_prompt, user_prompt),
            model=self._model,
            temperature=opts.get("temperature", 0.8),
            max_tokens=opts.get("max_tokens", 2048),
        )
        wall_start = time.monotonic()
        result = _run_async(self._backend.complete(request))
        wall_elapsed = time.monotonic() - wall_start
        timing = _extract_timing(result.usage, wall_elapsed)
        return result.text, timing


class GeminiCLIClient:
    """Benchmark client wrapping text-game-engine's GeminiCLIBackend."""

    def __init__(self, model: str | None = None, **kwargs: Any) -> None:
        from text_game_engine.backends.gemini_cli import GeminiCLIBackend
        self._model = model
        self._backend = GeminiCLIBackend(model=model, **kwargs)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        **opts: Any,
    ) -> tuple[str, TimingData]:
        request = CompletionRequest(
            messages=_build_messages(system_prompt, user_prompt),
            model=self._model,
            temperature=opts.get("temperature", 0.8),
            max_tokens=opts.get("max_tokens", 2048),
        )
        wall_start = time.monotonic()
        result = _run_async(self._backend.complete(request))
        wall_elapsed = time.monotonic() - wall_start
        timing = _extract_timing(result.usage, wall_elapsed)
        return result.text, timing


class CodexCLIClient:
    """Benchmark client wrapping text-game-engine's CodexCLIBackend."""

    def __init__(self, model: str | None = None, **kwargs: Any) -> None:
        from text_game_engine.backends.codex_cli import CodexCLIBackend
        self._model = model
        self._backend = CodexCLIBackend(model=model, **kwargs)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        **opts: Any,
    ) -> tuple[str, TimingData]:
        request = CompletionRequest(
            messages=_build_messages(system_prompt, user_prompt),
            model=self._model,
            temperature=opts.get("temperature", 0.8),
            max_tokens=opts.get("max_tokens", 2048),
        )
        wall_start = time.monotonic()
        result = _run_async(self._backend.complete(request))
        wall_elapsed = time.monotonic() - wall_start
        timing = _extract_timing(result.usage, wall_elapsed)
        return result.text, timing


class OpenCodeCLIClient:
    """Benchmark client wrapping text-game-engine's OpenCodeCLIBackend."""

    def __init__(self, model: str | None = None, **kwargs: Any) -> None:
        from text_game_engine.backends.opencode_cli import OpenCodeCLIBackend
        self._model = model
        self._backend = OpenCodeCLIBackend(model=model, **kwargs)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        **opts: Any,
    ) -> tuple[str, TimingData]:
        request = CompletionRequest(
            messages=_build_messages(system_prompt, user_prompt),
            model=self._model,
            temperature=opts.get("temperature", 0.8),
            max_tokens=opts.get("max_tokens", 2048),
        )
        wall_start = time.monotonic()
        result = _run_async(self._backend.complete(request))
        wall_elapsed = time.monotonic() - wall_start
        timing = _extract_timing(result.usage, wall_elapsed)
        return result.text, timing
