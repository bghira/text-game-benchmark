"""Tool usage checks: tool_called, tool_format_valid."""

from __future__ import annotations

from typing import Any

from tgb.checks.base import CheckResult
from tgb.config import Scenario, TurnSpec
from tgb.prompt_builder import AccumulatedState
from tgb.response_parser import ParsedResponse


def check_tool_called(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that a specific tool was invoked.

    Params:
        tool_name: str — expected tool_call value (e.g. "memory_search")
    """
    tool_name = params.get("tool_name", "")
    if not parsed.is_tool_call:
        return CheckResult(
            check_id="tool_called",
            passed=False,
            detail=f"Expected tool_call '{tool_name}' but response is not a tool call",
            category="tool_usage",
        )

    actual_tool = parsed.parsed_json.get("tool_call", "")
    if tool_name and actual_tool != tool_name:
        return CheckResult(
            check_id="tool_called",
            passed=False,
            detail=f"Expected tool '{tool_name}', got '{actual_tool}'",
            category="tool_usage",
        )

    return CheckResult(
        check_id="tool_called",
        passed=True,
        detail=f"Tool '{actual_tool}' called",
        category="tool_usage",
    )


def check_tool_format_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that tool call JSON has required keys.

    Params:
        required_keys: list[str] — keys expected alongside tool_call
    """
    if not parsed.is_tool_call:
        return CheckResult(
            check_id="tool_format_valid",
            passed=True,
            detail="Not a tool call — format check skipped",
            category="tool_usage",
        )

    data = parsed.parsed_json
    tool_name = data.get("tool_call", "")

    # Default required keys per tool type
    default_keys_map: dict[str, list[str]] = {
        "memory_search": ["queries"],
        "memory_store": ["category", "term", "memory"],
        "memory_terms": ["wildcard"],
        "memory_turn": ["turn_id"],
        "sms_write": ["thread", "from", "to", "message"],
        "sms_read": ["thread"],
        "sms_list": [],
        "sms_schedule": ["thread", "from", "to", "message", "delay_seconds"],
        "name_generate": [],
        "source_browse": [],
        "recent_turns": ["player_slugs", "npc_slugs"],
        "ready_to_write": [],
    }

    required_keys = params.get("required_keys", default_keys_map.get(tool_name, []))
    missing = [k for k in required_keys if k not in data]

    if missing:
        return CheckResult(
            check_id="tool_format_valid",
            passed=False,
            detail=f"Tool '{tool_name}' missing keys: {missing}",
            category="tool_usage",
        )

    # Check no extra keys besides tool_call + expected + known optional
    optional_keys = {"category", "limit"}
    allowed = {"tool_call"} | optional_keys | set(required_keys)
    extra = [k for k in data if k not in allowed]
    if extra:
        return CheckResult(
            check_id="tool_format_valid",
            passed=False,
            detail=f"Tool '{tool_name}' has unexpected keys: {extra}",
            category="tool_usage",
        )

    return CheckResult(
        check_id="tool_format_valid",
        passed=True,
        detail=f"Tool '{tool_name}' format valid",
        category="tool_usage",
    )
