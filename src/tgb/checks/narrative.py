"""Narrative checks: reasoning_concise, narration_length, narration_no_recap, no_inventory, no_markdown."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from tgb.checks.base import CheckResult
from tgb.checks.limits import (
    NARRATION_MAX_CHARS, NARRATION_MIN_CHARS,
    NARRATION_MAX_WORDS, NARRATION_MIN_WORDS,
    REASONING_MAX_CHARS,
)
from tgb.config import Scenario, TurnSpec
from tgb.prompt_builder import AccumulatedState
from tgb.response_parser import ParsedResponse


def check_reasoning_concise(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that reasoning is <= 1200 chars."""
    reasoning = parsed.parsed_json.get("reasoning", "")
    max_chars = params.get("max_chars", REASONING_MAX_CHARS)
    if not isinstance(reasoning, str):
        return CheckResult(
            check_id="reasoning_concise",
            passed=False,
            detail=f"reasoning is not a string",
            category="narrative",
        )
    length = len(reasoning)
    if length > max_chars:
        return CheckResult(
            check_id="reasoning_concise",
            passed=False,
            detail=f"reasoning is {length} chars (max {max_chars})",
            category="narrative",
        )
    return CheckResult(
        check_id="reasoning_concise",
        passed=True,
        detail=f"reasoning is {length} chars",
        category="narrative",
    )


def check_narration_length(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that narration is within char/word bounds."""
    narration = parsed.parsed_json.get("narration", "")
    if not isinstance(narration, str):
        return CheckResult(
            check_id="narration_length",
            passed=False,
            detail="narration is not a string",
            category="narrative",
        )

    max_chars = params.get("max_chars", NARRATION_MAX_CHARS)
    min_chars = params.get("min_chars", NARRATION_MIN_CHARS)
    max_words = params.get("max_words", NARRATION_MAX_WORDS)
    min_words = params.get("min_words", NARRATION_MIN_WORDS)

    char_len = len(narration)
    word_count = len(narration.split())

    issues = []
    if char_len > max_chars:
        issues.append(f"{char_len} chars > {max_chars} max")
    if char_len < min_chars:
        issues.append(f"{char_len} chars < {min_chars} min")
    if word_count > max_words:
        issues.append(f"{word_count} words > {max_words} max")
    if word_count < min_words:
        issues.append(f"{word_count} words < {min_words} min")

    if issues:
        return CheckResult(
            check_id="narration_length",
            passed=False,
            detail="; ".join(issues),
            category="narrative",
        )
    return CheckResult(
        check_id="narration_length",
        passed=True,
        detail=f"{char_len} chars, {word_count} words",
        category="narrative",
    )


def _ngrams(text: str, n: int) -> list[tuple[str, ...]]:
    """Extract n-grams from text."""
    words = re.findall(r"\w+", text.lower())
    return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]


def check_narration_no_recap(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that 4-gram overlap with prior narration is below threshold."""
    narration = parsed.parsed_json.get("narration", "")
    if not isinstance(narration, str) or not narration.strip():
        return CheckResult(
            check_id="narration_no_recap",
            passed=True,
            detail="No narration to check",
            category="narrative",
        )

    prior = state.last_narration
    if not prior:
        return CheckResult(
            check_id="narration_no_recap",
            passed=True,
            detail="No prior narration to compare",
            category="narrative",
        )

    n = params.get("ngram_size", 4)
    threshold = params.get("threshold", 0.3)

    current_ngrams = _ngrams(narration, n)
    prior_ngrams = set(_ngrams(prior, n))

    if not current_ngrams:
        return CheckResult(
            check_id="narration_no_recap",
            passed=True,
            detail="Narration too short for n-gram analysis",
            category="narrative",
        )

    overlap_count = sum(1 for ng in current_ngrams if ng in prior_ngrams)
    overlap_ratio = overlap_count / len(current_ngrams)

    if overlap_ratio > threshold:
        return CheckResult(
            check_id="narration_no_recap",
            passed=False,
            detail=f"{overlap_ratio:.0%} 4-gram overlap with prior narration (threshold {threshold:.0%})",
            category="narrative",
        )
    return CheckResult(
        check_id="narration_no_recap",
        passed=True,
        detail=f"{overlap_ratio:.0%} 4-gram overlap",
        category="narrative",
    )


def check_no_inventory_in_narration(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that narration doesn't contain inventory listings."""
    narration = parsed.parsed_json.get("narration", "")
    if not isinstance(narration, str):
        return CheckResult(
            check_id="no_inventory_in_narration",
            passed=True,
            detail="No narration",
            category="narrative",
        )

    # Patterns that indicate inventory listings
    inventory_patterns = [
        r"(?i)\binventory\s*:",
        r"(?i)\byou are carrying\b",
        r"(?i)\byou have\s*:\s*\n",
        r"(?i)\bitems?\s*:\s*\n",
        r"(?i)\bin your (pack|bag|backpack|satchel|pouch)\s*:",
    ]

    for pattern in inventory_patterns:
        match = re.search(pattern, narration)
        if match:
            return CheckResult(
                check_id="no_inventory_in_narration",
                passed=False,
                detail=f"Inventory listing found: '{match.group()}'",
                category="narrative",
            )
    return CheckResult(
        check_id="no_inventory_in_narration",
        passed=True,
        detail="No inventory listings in narration",
        category="narrative",
    )


def check_no_markdown_in_response(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that response doesn't contain markdown code fences."""
    raw = parsed.raw
    if "```" in raw:
        return CheckResult(
            check_id="no_markdown_in_response",
            passed=False,
            detail="Markdown code fences found in response",
            category="narrative",
        )
    return CheckResult(
        check_id="no_markdown_in_response",
        passed=True,
        detail="No markdown code fences",
        category="narrative",
    )
