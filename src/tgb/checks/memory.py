"""Memory tool checks — validates memory_search, memory_store, memory_terms, memory_turn."""

from __future__ import annotations

from typing import Any

from tgb.checks.base import CheckResult
from tgb.checks.limits import (
    MEMORY_SEARCH_MAX_QUERIES,
    MEMORY_SEARCH_CONTEXT_LINES_MAX,
    MEMORY_STORE_MEMORY_MAX_CHARS,
    MEMORY_TURN_ID_MIN,
)
from tgb.config import Scenario, TurnSpec
from tgb.prompt_builder import AccumulatedState
from tgb.response_parser import ParsedResponse


def check_memory_search_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Validate memory_search tool call fields.

    Engine expects:
    - queries: list of strings (or single string coerced to list), max 4, each non-empty
    - category (optional): string
    - before_lines (optional): int, 0–50
    - after_lines (optional): int, 0–50
    """
    if parsed.parsed_json.get("tool_call") != "memory_search":
        return CheckResult(
            check_id="memory_search_valid",
            passed=True,
            detail="Not a memory_search tool call, skipped",
            category="tool_usage",
        )

    data = parsed.parsed_json
    issues: list[str] = []

    # queries validation
    queries = data.get("queries")
    if queries is None:
        issues.append("Missing 'queries' field")
    else:
        # Coerce string to list
        if isinstance(queries, str):
            queries = [queries]
        if not isinstance(queries, list):
            issues.append(f"'queries' must be a string or list, got {type(queries).__name__}")
        else:
            if len(queries) == 0:
                issues.append("'queries' is empty")
            elif len(queries) > MEMORY_SEARCH_MAX_QUERIES:
                issues.append(
                    f"Too many queries: {len(queries)} (max {MEMORY_SEARCH_MAX_QUERIES})"
                )
            for i, q in enumerate(queries):
                if not isinstance(q, str):
                    issues.append(f"queries[{i}] is not a string")
                elif not q.strip():
                    issues.append(f"queries[{i}] is empty")

    # category validation (optional)
    category = data.get("category")
    if category is not None and not isinstance(category, str):
        issues.append(f"'category' must be a string, got {type(category).__name__}")

    # before_lines validation (optional)
    before_lines = data.get("before_lines")
    if before_lines is not None:
        if isinstance(before_lines, bool) or not isinstance(before_lines, int):
            issues.append(f"'before_lines' must be an integer, got {type(before_lines).__name__}")
        elif before_lines < 0 or before_lines > MEMORY_SEARCH_CONTEXT_LINES_MAX:
            issues.append(
                f"'before_lines' {before_lines} out of range [0, {MEMORY_SEARCH_CONTEXT_LINES_MAX}]"
            )

    # after_lines validation (optional)
    after_lines = data.get("after_lines")
    if after_lines is not None:
        if isinstance(after_lines, bool) or not isinstance(after_lines, int):
            issues.append(f"'after_lines' must be an integer, got {type(after_lines).__name__}")
        elif after_lines < 0 or after_lines > MEMORY_SEARCH_CONTEXT_LINES_MAX:
            issues.append(
                f"'after_lines' {after_lines} out of range [0, {MEMORY_SEARCH_CONTEXT_LINES_MAX}]"
            )

    if issues:
        return CheckResult(
            check_id="memory_search_valid",
            passed=False,
            detail="; ".join(issues),
            category="tool_usage",
        )
    return CheckResult(
        check_id="memory_search_valid",
        passed=True,
        detail="memory_search fields valid",
        category="tool_usage",
    )


def check_memory_store_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Validate memory_store tool call fields.

    Engine expects:
    - category: required, non-empty string
    - memory: required, non-empty string, ≤1600 chars
    - term (optional): string if present
    """
    if parsed.parsed_json.get("tool_call") != "memory_store":
        return CheckResult(
            check_id="memory_store_valid",
            passed=True,
            detail="Not a memory_store tool call, skipped",
            category="tool_usage",
        )

    data = parsed.parsed_json
    issues: list[str] = []

    # category validation (required)
    category = data.get("category")
    if category is None or not isinstance(category, str) or not category.strip():
        issues.append("Missing or empty 'category' field")

    # memory validation (required)
    memory = data.get("memory")
    if memory is None or not isinstance(memory, str) or not memory.strip():
        issues.append("Missing or empty 'memory' field")
    elif len(memory) > MEMORY_STORE_MEMORY_MAX_CHARS:
        issues.append(
            f"'memory' is {len(memory)} chars (max {MEMORY_STORE_MEMORY_MAX_CHARS})"
        )

    # term validation (optional)
    term = data.get("term")
    if term is not None and not isinstance(term, str):
        issues.append(f"'term' must be a string, got {type(term).__name__}")

    if issues:
        return CheckResult(
            check_id="memory_store_valid",
            passed=False,
            detail="; ".join(issues),
            category="tool_usage",
        )
    return CheckResult(
        check_id="memory_store_valid",
        passed=True,
        detail="memory_store fields valid",
        category="tool_usage",
    )


def check_memory_terms_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Validate memory_terms tool call fields.

    Engine expects:
    - wildcard (optional): string (engine defaults to "*" when absent)
    """
    if parsed.parsed_json.get("tool_call") != "memory_terms":
        return CheckResult(
            check_id="memory_terms_valid",
            passed=True,
            detail="Not a memory_terms tool call, skipped",
            category="tool_usage",
        )

    data = parsed.parsed_json
    issues: list[str] = []

    # wildcard validation (optional)
    wildcard = data.get("wildcard")
    if wildcard is not None and not isinstance(wildcard, str):
        issues.append(f"'wildcard' must be a string, got {type(wildcard).__name__}")

    if issues:
        return CheckResult(
            check_id="memory_terms_valid",
            passed=False,
            detail="; ".join(issues),
            category="tool_usage",
        )
    return CheckResult(
        check_id="memory_terms_valid",
        passed=True,
        detail="memory_terms fields valid",
        category="tool_usage",
    )


def check_memory_turn_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Validate memory_turn tool call fields.

    Engine expects:
    - turn_id: required, integer (not bool), ≥1
    """
    if parsed.parsed_json.get("tool_call") != "memory_turn":
        return CheckResult(
            check_id="memory_turn_valid",
            passed=True,
            detail="Not a memory_turn tool call, skipped",
            category="tool_usage",
        )

    data = parsed.parsed_json
    issues: list[str] = []

    # turn_id validation (required)
    turn_id = data.get("turn_id")
    if turn_id is None:
        issues.append("Missing 'turn_id' field")
    elif isinstance(turn_id, bool) or not isinstance(turn_id, int):
        issues.append(f"'turn_id' must be an integer, got {type(turn_id).__name__}")
    elif turn_id < MEMORY_TURN_ID_MIN:
        issues.append(f"'turn_id' {turn_id} must be >= {MEMORY_TURN_ID_MIN}")

    if issues:
        return CheckResult(
            check_id="memory_turn_valid",
            passed=False,
            detail="; ".join(issues),
            category="tool_usage",
        )
    return CheckResult(
        check_id="memory_turn_valid",
        passed=True,
        detail="memory_turn fields valid",
        category="tool_usage",
    )
