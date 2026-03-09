"""JSON structure checks: json_valid, json_keys_present, json_types_correct, xp_range, reasoning_present."""

from __future__ import annotations

from typing import Any

from tgb.checks.base import CheckResult
from tgb.checks.limits import XP_MIN, XP_MAX
from tgb.config import Scenario, TurnSpec
from tgb.prompt_builder import AccumulatedState
from tgb.response_parser import ParsedResponse

# Default required keys in a final (non-tool-call) response
DEFAULT_REQUIRED_KEYS = [
    "reasoning", "narration", "state_update", "summary_update", "xp_awarded",
]

# Expected types for standard response keys
EXPECTED_TYPES: dict[str, type | tuple[type, ...]] = {
    "reasoning": str,
    "narration": str,
    "state_update": dict,
    "summary_update": str,
    "xp_awarded": (int, float),
    "player_state_update": dict,
    "scene_image_prompt": str,
    "character_updates": dict,
    "give_item": dict,
    "turn_visibility": dict,
    "calendar_update": dict,
    "dice_check": dict,
    "puzzle_trigger": dict,
    "minigame_challenge": dict,
    "set_timer_delay": (int, float),
    "set_timer_event": str,
    "set_timer_interruptible": bool,
    "set_timer_interrupt_action": (str, type(None)),
    "set_timer_interrupt_scope": str,
}


def check_json_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that the raw text parses as valid JSON."""
    if parsed.parse_error:
        return CheckResult(
            check_id="json_valid",
            passed=False,
            detail=f"JSON parse error: {parsed.parse_error}",
            category="json_structure",
        )
    if not parsed.parsed_json:
        return CheckResult(
            check_id="json_valid",
            passed=False,
            detail="Response parsed to empty dict",
            category="json_structure",
        )
    return CheckResult(
        check_id="json_valid",
        passed=True,
        detail="Valid JSON object",
        category="json_structure",
    )


def check_json_keys_present(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that required keys exist in the parsed JSON."""
    if parsed.is_tool_call:
        return CheckResult(
            check_id="json_keys_present",
            passed=True,
            detail="Tool call response — keys check skipped",
            category="json_structure",
        )

    required = params.get("keys", DEFAULT_REQUIRED_KEYS)
    data = parsed.parsed_json
    missing = [k for k in required if k not in data]

    if missing:
        return CheckResult(
            check_id="json_keys_present",
            passed=False,
            detail=f"Missing keys: {missing}",
            category="json_structure",
        )
    return CheckResult(
        check_id="json_keys_present",
        passed=True,
        detail=f"All {len(required)} required keys present",
        category="json_structure",
    )


def check_json_types_correct(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that value types match the expected schema."""
    if parsed.is_tool_call:
        return CheckResult(
            check_id="json_types_correct",
            passed=True,
            detail="Tool call response — type check skipped",
            category="json_structure",
        )

    data = parsed.parsed_json
    wrong = []
    for key, expected in EXPECTED_TYPES.items():
        if key in data:
            val = data[key]
            if val is not None and not isinstance(val, expected):
                wrong.append(f"{key}: expected {expected}, got {type(val).__name__}")

    if wrong:
        return CheckResult(
            check_id="json_types_correct",
            passed=False,
            detail=f"Type mismatches: {'; '.join(wrong)}",
            category="json_structure",
        )
    return CheckResult(
        check_id="json_types_correct",
        passed=True,
        detail="All present keys have correct types",
        category="json_structure",
    )


def check_xp_range(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that xp_awarded is an int in [0, 10]."""
    data = parsed.parsed_json
    xp = data.get("xp_awarded")
    if xp is None:
        return CheckResult(
            check_id="xp_range",
            passed=False,
            detail="xp_awarded missing",
            category="json_structure",
        )
    if isinstance(xp, bool) or not isinstance(xp, (int, float)):
        return CheckResult(
            check_id="xp_range",
            passed=False,
            detail=f"xp_awarded is {type(xp).__name__}, not int",
            category="json_structure",
        )
    if isinstance(xp, float) and xp != int(xp):
        return CheckResult(
            check_id="xp_range",
            passed=False,
            detail=f"xp_awarded is {xp} (float), must be an integer",
            category="json_structure",
        )
    xp_int = int(xp)
    min_xp = params.get("min", XP_MIN)
    max_xp = params.get("max", XP_MAX)
    if not (min_xp <= xp_int <= max_xp):
        return CheckResult(
            check_id="xp_range",
            passed=False,
            detail=f"xp_awarded={xp_int} outside [{min_xp}, {max_xp}]",
            category="json_structure",
        )
    return CheckResult(
        check_id="xp_range",
        passed=True,
        detail=f"xp_awarded={xp_int} in range",
        category="json_structure",
    )


def check_reasoning_present(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that reasoning key exists and is non-empty."""
    data = parsed.parsed_json
    reasoning = data.get("reasoning")
    if not reasoning or not isinstance(reasoning, str) or not reasoning.strip():
        return CheckResult(
            check_id="reasoning_present",
            passed=False,
            detail="reasoning is missing or empty",
            category="json_structure",
        )
    return CheckResult(
        check_id="reasoning_present",
        passed=True,
        detail=f"reasoning present ({len(reasoning)} chars)",
        category="json_structure",
    )
