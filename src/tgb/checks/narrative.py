"""Narrative checks: reasoning_concise, narration_length, narration_no_recap, no_inventory, no_markdown, anti-echo, therapist-speak, abstract-summary."""

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


# ── Writing craft checks (mirrors engine WRITING_CRAFT + ANTI-ECHO rules) ──

# ANTI-ECHO: narration should not restate/mirror the player's action wording
_ECHO_NGRAM_SIZE = 4
_ECHO_THRESHOLD_DEFAULT = 0.35


def check_narration_no_echo(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that narration doesn't echo the player's action wording.

    Mirrors the engine's ANTI-ECHO directive: 'do NOT restate, paraphrase,
    or mirror the player's just-written wording.'
    Uses n-gram overlap between the player's action text and the first
    sentence(s) of narration.
    """
    narration = parsed.parsed_json.get("narration", "")
    if not isinstance(narration, str) or not narration.strip():
        return CheckResult(
            check_id="narration_no_echo",
            passed=True,
            detail="No narration to check",
            category="narrative",
        )

    action = turn.action
    if not action or not action.strip():
        return CheckResult(
            check_id="narration_no_echo",
            passed=True,
            detail="No action to compare",
            category="narrative",
        )

    # Compare action against first ~200 chars of narration (the opening)
    opening = narration[:200]

    n = params.get("ngram_size", _ECHO_NGRAM_SIZE)
    threshold = params.get("threshold", _ECHO_THRESHOLD_DEFAULT)

    action_ngrams = set(_ngrams(action, n))
    if not action_ngrams:
        return CheckResult(
            check_id="narration_no_echo",
            passed=True,
            detail="Action too short for n-gram analysis",
            category="narrative",
        )

    opening_ngrams = _ngrams(opening, n)
    if not opening_ngrams:
        return CheckResult(
            check_id="narration_no_echo",
            passed=True,
            detail="Narration opening too short for n-gram analysis",
            category="narrative",
        )

    echoed = sum(1 for ng in opening_ngrams if ng in action_ngrams)
    ratio = echoed / len(opening_ngrams)

    if ratio > threshold:
        return CheckResult(
            check_id="narration_no_echo",
            passed=False,
            detail=f"{ratio:.0%} echo overlap with player action (threshold {threshold:.0%})",
            category="narrative",
        )
    return CheckResult(
        check_id="narration_no_echo",
        passed=True,
        detail=f"{ratio:.0%} echo overlap",
        category="narrative",
    )


# Therapist-speak / contrived emotional shorthand detection
_THERAPIST_SPEAK_PHRASES = [
    r"\bhold(?:ing)?\s+space\b",
    r"\bbe\s+present\b",
    r"\bshow(?:ing)?\s+up\b(?:\s+for\b)",
    r"\bdo(?:ing)?\s+the\s+work\b",
    r"\blean(?:ing)?\s+into?\b",
    r"\bsit(?:ting)?\s+with\b(?:\s+(?:that|this|the|it)\b)",
    r"\bunpack(?:ing)?\b(?:\s+(?:that|this|the|it)\b)",
    r"\bprocess(?:ing)?\b(?:\s+(?:that|this|the|it)\b)",
    r"\bfeel(?:ing)?\s+seen\b",
    r"\bvalid(?:at(?:e|ing))?\s+(?:your|his|her|their)\b",
    r"\bsafe\s+space\b",
    r"\bset(?:ting)?\s+(?:a\s+)?boundar(?:y|ies)\b",
    r"\bheal(?:ing)?\s+journey\b",
    r"\bemotional\s+labor\b",
]

_THERAPIST_THRESHOLD_DEFAULT = 2  # number of distinct phrases before flagging


def check_narration_no_therapist_speak(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that narration avoids contrived therapeutic language.

    Mirrors the engine's anti-pattern: 'Avoid contrived emotional-summary
    language or therapist-speak unless that exact voice is canonically right
    for the speaking character.'
    """
    narration = parsed.parsed_json.get("narration", "")
    if not isinstance(narration, str) or not narration.strip():
        return CheckResult(
            check_id="narration_no_therapist_speak",
            passed=True,
            detail="No narration to check",
            category="narrative",
        )

    threshold = params.get("threshold", _THERAPIST_THRESHOLD_DEFAULT)
    hits: list[str] = []
    for pattern in _THERAPIST_SPEAK_PHRASES:
        match = re.search(pattern, narration, re.IGNORECASE)
        if match:
            hits.append(match.group())

    if len(hits) >= threshold:
        return CheckResult(
            check_id="narration_no_therapist_speak",
            passed=False,
            detail=f"Therapist-speak detected ({len(hits)} phrases): {hits}",
            category="narrative",
        )
    return CheckResult(
        check_id="narration_no_therapist_speak",
        passed=True,
        detail=f"No therapist-speak pattern ({len(hits)} matches)" if hits else "Clean",
        category="narrative",
    )


# Abstract-summary detection — WRITING_CRAFT: "Abstract summary is not narration"
_ABSTRACT_SUMMARY_PATTERNS = [
    r"(?i)\bthey?\s+(discussed|talked\s+about|went\s+over|covered)\s+(?:various|several|many|a\s+(?:number|range|variety)\s+of)\b",
    r"(?i)\btime\s+passed\b.*\bthey\b.*\b(continued|went\s+on|kept)\b",
    r"(?i)\bthe\s+conversation\s+(continued|went\s+on|flowed|turned)\b",
    r"(?i)\bafter\s+(?:some|much|a\s+(?:long|brief))\s+(?:discussion|deliberation|conversation|debate)\b",
    r"(?i)\b(?:various|several|many)\s+(?:topics|subjects|matters|issues)\s+(?:were|came\s+up)\b",
]

_ABSTRACT_THRESHOLD_DEFAULT = 2


def check_narration_not_abstract(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that narration uses concrete detail, not abstract summarization.

    Mirrors WRITING_CRAFT: 'Ground every sentence in the concrete: sensory
    detail, specific objects, named places. Abstract summary is not narration.'
    """
    narration = parsed.parsed_json.get("narration", "")
    if not isinstance(narration, str) or not narration.strip():
        return CheckResult(
            check_id="narration_not_abstract",
            passed=True,
            detail="No narration to check",
            category="narrative",
        )

    threshold = params.get("threshold", _ABSTRACT_THRESHOLD_DEFAULT)
    hits: list[str] = []
    for pattern in _ABSTRACT_SUMMARY_PATTERNS:
        match = re.search(pattern, narration)
        if match:
            hits.append(match.group())

    if len(hits) >= threshold:
        return CheckResult(
            check_id="narration_not_abstract",
            passed=False,
            detail=f"Abstract summary patterns detected ({len(hits)}): {hits}",
            category="narrative",
        )
    return CheckResult(
        check_id="narration_not_abstract",
        passed=True,
        detail=f"Narration is concrete ({len(hits)} abstract patterns)" if hits else "Concrete narration",
        category="narrative",
    )


# ── Emotional register sustain ────────────────────────────────

_EMOTION_MARKERS: list[str] = [
    r"\btears\b",
    r"\bembrace[ds]?\b",
    r"\bwhisper(?:s|ed|ing)?\b",
    r"\bsilence\s+between\b",
    r"\bvoice\s+crack(?:s|ed|ing)?\b",
    r"\bheld\s+(?:her|his|their|your)?\s*hand\b",
    r"\bvulnerab(?:le|ility)\b",
    r"\bconfess(?:es|ed|ion|ions)?\b",
    r"\bsobb(?:s|ed|ing)?\b",
    r"\beyes?\s+glisten(?:s|ed|ing)?\b",
    r"\bpull(?:s|ed|ing)?\s+close\b",
    r"\bbreath\s+c(?:aught|atches)\b",
    r"\btrembl(?:e[ds]?|ing)\b",
    r"\bchok(?:e[ds]?|ing)\s+(?:on|back)\b",
    r"\bforgive\b",
    r"\bgrief\b",
    r"\bheartbreak\b",
    r"\baching\b",
    r"\blonging\b",
]

_LOGISTICS_MARKERS: list[str] = [
    r"\bexits?\b(?:\s*:|\s+are\b|\s+lead\b)",
    r"\bfrom\s+here\s+you\s+can\b",
    r"\byou\s+notice\s+a\s+door\b",
    r"\bdirections?\b(?:\s*:|\s+to\b)",
    r"\boptions?\b(?:\s*:|\s+are\b|\s+include\b)",
    r"\b(?:north|south|east|west)\b",
    r"\byou\s+could\s+go\b",
    r"\binventory\b",
    r"\bthe\s+room\s+contains\b",
    r"\bwhat\s+(?:do\s+you|will\s+you|would\s+you)\b",
    r"\bwhere\s+will\s+you\b",
    r"\byou\s+(?:can|could)\s+(?:also\s+)?(?:head|travel|walk|go)\b",
    r"\bavailable\s+(?:paths?|exits?|routes?)\b",
]


def check_narration_no_logistics_after_emotion(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that narration doesn't pivot from emotional register to logistics.

    Splits narration at midpoint. If the first half has >=N emotional markers
    AND the second half has >=M logistics markers, the turn breaks register.

    Params:
        emotion_threshold: int — min emotional markers in first half (default 2)
        logistics_threshold: int — min logistics markers in second half (default 2)
    """
    narration = parsed.parsed_json.get("narration", "")
    if not isinstance(narration, str) or not narration.strip():
        return CheckResult(
            check_id="narration_no_logistics_after_emotion",
            passed=True,
            detail="No narration to check",
            category="narrative",
        )

    emotion_threshold = params.get("emotion_threshold", 2)
    logistics_threshold = params.get("logistics_threshold", 2)

    mid = len(narration) // 2
    first_half = narration[:mid]
    second_half = narration[mid:]

    emotion_hits = []
    for pattern in _EMOTION_MARKERS:
        match = re.search(pattern, first_half, re.IGNORECASE)
        if match:
            emotion_hits.append(match.group())

    logistics_hits = []
    for pattern in _LOGISTICS_MARKERS:
        match = re.search(pattern, second_half, re.IGNORECASE)
        if match:
            logistics_hits.append(match.group())

    if len(emotion_hits) >= emotion_threshold and len(logistics_hits) >= logistics_threshold:
        return CheckResult(
            check_id="narration_no_logistics_after_emotion",
            passed=False,
            detail=(
                f"Emotional register broken: {len(emotion_hits)} emotion markers "
                f"in first half ({emotion_hits[:3]}), then {len(logistics_hits)} "
                f"logistics markers in second half ({logistics_hits[:3]})"
            ),
            category="narrative",
        )

    return CheckResult(
        check_id="narration_no_logistics_after_emotion",
        passed=True,
        detail=(
            f"Register sustained ({len(emotion_hits)} emotion, "
            f"{len(logistics_hits)} logistics)"
        ),
        category="narrative",
    )


# ── Filing-cabinet phrasing ban ──────────────────────────

_FILING_CABINET_PATTERNS = [
    r"(?i)\bfil(?:e[ds]?|ing)\s+(?:that|this|it|the\s+\w+)\s+(?:away|for\s+later|in\s+(?:her|his|their)\s+(?:mind|memory|mental))\b",
    r"(?i)\bstor(?:e[ds]?|ing)\s+(?:that|this|it|the\s+\w+)\s+(?:away|for\s+later)\b",
    r"(?i)\btuck(?:s|ed|ing)?\s+(?:that|this|it|the\s+\w+)\s+away\s+(?:for|in\s+(?:her|his|their))\b",
    r"(?i)\bcatalog(?:u?e[ds]?|u?ing)?\s+(?:that|this|it|the\s+\w+)\s+(?:mentally|in\s+(?:her|his|their)\s+mind)\b",
    r"(?i)\bfil(?:e[ds]?|ing)\s+(?:facts?|information|detail|data)\b",
    r"(?i)\bment(?:al(?:ly)?\s+)?not(?:e[ds]?|ing)\s+(?:that|this|it)\b",
    r"(?i)\badd(?:s|ed|ing)?\s+(?:that|this|it)\s+to\s+(?:her|his|their)\s+mental\b",
    r"(?i)\bprocess(?:es|ed|ing)?\s+(?:the|this|that)\s+(?:new\s+)?(?:information|data|intel|detail)\b",
]


def check_narration_no_filing_cabinet(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that narration avoids filing-cabinet phrasing for character reactions.

    Engine BAN: Characters should not 'file facts away', 'store that for later',
    or otherwise process new information like clerks or databases. If someone
    registers something important, describe a fresher concrete reaction instead.
    """
    narration = parsed.parsed_json.get("narration", "")
    if not isinstance(narration, str) or not narration.strip():
        return CheckResult(
            check_id="narration_no_filing_cabinet",
            passed=True,
            detail="No narration to check",
            category="narrative",
        )

    hits: list[str] = []
    for pattern in _FILING_CABINET_PATTERNS:
        match = re.search(pattern, narration)
        if match:
            hits.append(match.group())

    if hits:
        return CheckResult(
            check_id="narration_no_filing_cabinet",
            passed=False,
            detail=f"Filing-cabinet phrasing detected: {hits}",
            category="narrative",
        )
    return CheckResult(
        check_id="narration_no_filing_cabinet",
        passed=True,
        detail="No filing-cabinet phrasing",
        category="narrative",
    )
