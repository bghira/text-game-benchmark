"""Thin wrapper around text-game-engine's OllamaBackend for benchmark use."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from text_game_engine.backends.base import ChatMessage, CompletionRequest
from text_game_engine.backends.ollama import OllamaBackend


@dataclass(frozen=True)
class TimingData:
    """Timing and token count data from a completion."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    eval_tokens_per_sec: float = 0.0
    wall_clock_seconds: float = 0.0


class OllamaClient:
    """Benchmark client wrapping OllamaBackend."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        temperature: float = 0.8,
        max_tokens: int = 2048,
    ) -> None:
        self._model = model
        self._backend = OllamaBackend(model=model, base_url=base_url)
        self._temperature = temperature
        self._max_tokens = max_tokens

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        **opts: Any,
    ) -> tuple[str, TimingData]:
        """Run a completion and return (text, timing_data)."""
        messages = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=user_prompt))

        request = CompletionRequest(
            messages=messages,
            model=self._model,
            temperature=opts.get("temperature", self._temperature),
            max_tokens=opts.get("max_tokens", self._max_tokens),
        )

        wall_start = time.monotonic()
        result = asyncio.get_event_loop().run_until_complete(
            self._backend.complete(request)
        )
        wall_elapsed = time.monotonic() - wall_start

        timing = self._extract_timing(result.raw_response, wall_elapsed)
        return result.text, timing

    def complete_async(
        self,
        system_prompt: str,
        user_prompt: str,
        **opts: Any,
    ) -> tuple[str, TimingData]:
        """Async-compatible completion (creates event loop if needed)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in async context — use nest_asyncio pattern or thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(self.complete, system_prompt, user_prompt, **opts).result()
        return self.complete(system_prompt, user_prompt, **opts)

    def _extract_timing(self, raw: dict[str, Any] | None, wall_seconds: float) -> TimingData:
        """Extract timing data from ollama raw response."""
        if not raw:
            return TimingData(wall_clock_seconds=wall_seconds)

        prompt_tokens = raw.get("prompt_eval_count", 0)
        completion_tokens = raw.get("eval_count", 0)

        # Ollama reports eval_duration in nanoseconds
        eval_duration_ns = raw.get("eval_duration", 0)
        if eval_duration_ns and completion_tokens:
            eval_tps = completion_tokens / (eval_duration_ns / 1e9)
        else:
            eval_tps = 0.0

        return TimingData(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            eval_tokens_per_sec=round(eval_tps, 1),
            wall_clock_seconds=round(wall_seconds, 2),
        )
