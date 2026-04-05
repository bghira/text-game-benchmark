"""SMS checks — validates SMS tool usage against the engine contract.

The engine's SMS contract:
- sms_list: {"tool_call": "sms_list", "wildcard": "*"}
- sms_read: {"tool_call": "sms_read", "thread": "slug", "limit": 20}
- sms_write: {"tool_call": "sms_write", "thread": "slug", "from": "...", "to": "...", "message": "..."}
- sms_schedule: {"tool_call": "sms_schedule", "thread": "slug", "from": "...", "to": "...", "message": "...", "delay_seconds": N}

Critical rules:
- When an NPC replies via text/phone, sms_write MUST be called for the NPC reply BEFORE final narration
- Both sides of a conversation must be in the SMS log
- Do NOT leak scene context into SMS content unless the SMS explicitly mentions it
- NPC SMS knowledge must be limited to what that thread and continuity plausibly reveal
- Use stable contact thread slugs (always the same slug for both directions)
"""

from __future__ import annotations

import re
from typing import Any

from tgb.checks.base import CheckResult
from tgb.checks.limits import (
    SMS_FIELD_MAX_CHARS,
    SMS_MESSAGE_MAX_CHARS,
    SMS_DELAY_MIN,
    SMS_DELAY_MAX,
    SMS_READ_LIMIT_MIN,
    SMS_READ_LIMIT_MAX,
)
from tgb.config import Scenario, TurnSpec
from tgb.prompt_builder import AccumulatedState
from tgb.response_parser import ParsedResponse


def check_sms_tool_used(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that SMS tools are used when the player action involves phone/text/call.

    Params:
        expect_sms_tool: str — specific tool expected ("sms_read", "sms_write", "sms_list", "sms_schedule")
    """
    action_lower = turn.action.lower()
    action_involves_phone = bool(re.search(
        r"\b(?:text|sms|message|phone|call|ring|dial)\b",
        action_lower,
    ))

    expect_tool = params.get("expect_sms_tool", "")

    if parsed.is_tool_call:
        actual_tool = parsed.parsed_json.get("tool_call", "")
        if expect_tool and actual_tool != expect_tool:
            return CheckResult(
                check_id="sms_tool_used",
                passed=False,
                detail=f"Expected '{expect_tool}' but got '{actual_tool}'",
                category="sms",
            )
        if actual_tool.startswith("sms_"):
            return CheckResult(
                check_id="sms_tool_used",
                passed=True,
                detail=f"SMS tool '{actual_tool}' called correctly",
                category="sms",
            )

    if action_involves_phone and expect_tool:
        return CheckResult(
            check_id="sms_tool_used",
            passed=False,
            detail=f"Player action involves phone/text but no SMS tool call made (expected '{expect_tool}')",
            category="sms",
        )

    if action_involves_phone and not parsed.is_tool_call:
        return CheckResult(
            check_id="sms_tool_used",
            passed=False,
            detail="Player action involves phone/text but response is not a tool call",
            category="sms",
        )

    return CheckResult(
        check_id="sms_tool_used",
        passed=True,
        detail="SMS tool usage appropriate",
        category="sms",
    )


def check_sms_write_fields_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that sms_write tool calls have all required fields with valid values.

    Required: thread, from, to, message
    Limits: thread/from/to <= 80 chars, message <= 500 chars
    """
    if not parsed.is_tool_call:
        return CheckResult(
            check_id="sms_write_fields_valid",
            passed=True,
            detail="Not a tool call",
            category="sms",
        )

    data = parsed.parsed_json
    tool = data.get("tool_call", "")
    if tool not in ("sms_write", "sms_schedule"):
        return CheckResult(
            check_id="sms_write_fields_valid",
            passed=True,
            detail=f"Tool is '{tool}', not sms_write/sms_schedule",
            category="sms",
        )

    issues: list[str] = []

    # Required fields
    for field in ("thread", "from", "to", "message"):
        val = data.get(field)
        if not val or not isinstance(val, str) or not val.strip():
            issues.append(f"'{field}' missing or empty")

    # Length limits
    for field, limit in [
        ("thread", SMS_FIELD_MAX_CHARS),
        ("from", SMS_FIELD_MAX_CHARS),
        ("to", SMS_FIELD_MAX_CHARS),
        ("message", SMS_MESSAGE_MAX_CHARS),
    ]:
        val = data.get(field, "")
        if isinstance(val, str) and len(val) > limit:
            issues.append(f"'{field}' is {len(val)} chars (max {limit})")

    # sms_schedule requires delay_seconds
    if tool == "sms_schedule":
        delay = data.get("delay_seconds")
        if delay is None:
            issues.append("sms_schedule missing 'delay_seconds'")
        elif not isinstance(delay, (int, float)):
            issues.append(f"delay_seconds is {type(delay).__name__}, not int")
        else:
            delay_int = int(delay)
            if delay_int < SMS_DELAY_MIN or delay_int > SMS_DELAY_MAX:
                issues.append(f"delay_seconds={delay_int} outside [{SMS_DELAY_MIN}, {SMS_DELAY_MAX}]")

    if issues:
        return CheckResult(
            check_id="sms_write_fields_valid",
            passed=False,
            detail="; ".join(issues),
            category="sms",
        )
    return CheckResult(
        check_id="sms_write_fields_valid",
        passed=True,
        detail=f"{tool} fields valid",
        category="sms",
    )


def check_sms_both_sides_recorded(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that when narration shows an NPC replying via text, sms_write was called for both sides.

    This is a CRITICAL rule: if the model narrates an NPC texting back but doesn't
    sms_write the NPC's reply, the reply is lost permanently.

    This check examines the narration for signs of an NPC text reply and cross-references
    with whether a tool call was made. For multi-step tool usage, this is a heuristic —
    the scenario should be designed so the check fires on the final response turn.

    Params:
        npc_name: str — the NPC expected to reply
    """
    narration = parsed.parsed_json.get("narration", "")
    if not isinstance(narration, str) or parsed.is_tool_call:
        return CheckResult(
            check_id="sms_both_sides_recorded",
            passed=True,
            detail="Tool call turn or no narration — checked on final response",
            category="sms",
        )

    npc_name = params.get("npc_name", "")

    # Look for signs of NPC text reply in narration
    reply_patterns = [
        r"(?i)\b(?:texts?|messages?|replies|responds?|writes? back|buzzes?|pings?)\b.*(?:back|reply|response)",
        r"(?i)(?:phone|screen)\s+(?:lights up|buzzes|vibrates|pings)",
        r"(?i)(?:new|incoming)\s+(?:text|message|sms)",
        r"(?i)(?:reads?|shows?|displays?)\s*[:\"]",
    ]
    if npc_name:
        reply_patterns.append(rf"(?i)\b{re.escape(npc_name)}\b.*(?:texts?|messages?|replies|writes?)")

    npc_reply_narrated = any(re.search(p, narration) for p in reply_patterns)

    if not npc_reply_narrated:
        return CheckResult(
            check_id="sms_both_sides_recorded",
            passed=True,
            detail="No NPC text reply detected in narration",
            category="sms",
        )

    # NPC reply was narrated — check if this response (or accumulated state)
    # shows evidence of sms_write. Since we're checking the final response,
    # the model should have done sms_write in a prior tool turn.
    # If the narration mentions an NPC reply but we're in the final response
    # without evidence of recording, that's a failure.
    #
    # Heuristic: check accumulated state for SMS tracking evidence
    sms_state = state.campaign_state.get("_sms_threads")
    if sms_state is None:
        # No SMS state at all — the model didn't record anything
        return CheckResult(
            check_id="sms_both_sides_recorded",
            passed=False,
            detail=f"NPC text reply narrated but no SMS state exists — reply likely not recorded via sms_write",
            category="sms",
        )

    return CheckResult(
        check_id="sms_both_sides_recorded",
        passed=True,
        detail="NPC reply narrated and SMS state exists",
        category="sms",
    )


def check_sms_no_context_leak(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that SMS content doesn't leak scene context the NPC shouldn't know.

    NPC SMS responses must be limited to what the thread history and
    established continuity plausibly reveal.

    Params:
        forbidden_context: list[str] — terms/facts the NPC shouldn't know via SMS
    """
    if not parsed.is_tool_call:
        return CheckResult(
            check_id="sms_no_context_leak",
            passed=True,
            detail="Not a tool call",
            category="sms",
        )

    data = parsed.parsed_json
    tool = data.get("tool_call", "")
    if tool not in ("sms_write", "sms_schedule"):
        return CheckResult(
            check_id="sms_no_context_leak",
            passed=True,
            detail=f"Tool is '{tool}', not sms_write/sms_schedule",
            category="sms",
        )

    message = data.get("message", "")
    if not isinstance(message, str):
        return CheckResult(
            check_id="sms_no_context_leak",
            passed=True,
            detail="No message content",
            category="sms",
        )

    forbidden = params.get("forbidden_context", [])
    if not forbidden:
        return CheckResult(
            check_id="sms_no_context_leak",
            passed=True,
            detail="No forbidden context terms specified",
            category="sms",
        )

    message_lower = message.lower()
    leaks = [term for term in forbidden if term.lower() in message_lower]
    if leaks:
        return CheckResult(
            check_id="sms_no_context_leak",
            passed=False,
            detail=f"SMS leaks context NPC shouldn't know: {leaks}",
            category="sms",
        )
    return CheckResult(
        check_id="sms_no_context_leak",
        passed=True,
        detail="No forbidden context leaked in SMS",
        category="sms",
    )


def check_sms_thread_slug_stable(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that SMS thread slugs are stable and consistent.

    The engine requires a stable contact thread slug for both directions
    (e.g. always 'elizabeth' for Deshawn<->Elizabeth), not per-sender names.

    Params:
        expected_thread: str — the expected thread slug for this interaction
    """
    if not parsed.is_tool_call:
        return CheckResult(
            check_id="sms_thread_slug_stable",
            passed=True,
            detail="Not a tool call",
            category="sms",
        )

    data = parsed.parsed_json
    tool = data.get("tool_call", "")
    if not tool.startswith("sms_"):
        return CheckResult(
            check_id="sms_thread_slug_stable",
            passed=True,
            detail=f"Not an SMS tool",
            category="sms",
        )

    thread = data.get("thread", "")
    expected_thread = params.get("expected_thread", "")

    if expected_thread and thread != expected_thread:
        return CheckResult(
            check_id="sms_thread_slug_stable",
            passed=False,
            detail=f"Thread slug '{thread}' doesn't match expected '{expected_thread}'",
            category="sms",
        )

    # Check slug format (should be lowercase, stable)
    if thread and not re.match(r"^[a-z][a-z0-9_-]*$", thread):
        return CheckResult(
            check_id="sms_thread_slug_stable",
            passed=False,
            detail=f"Thread slug '{thread}' doesn't follow lowercase slug format",
            category="sms",
        )

    return CheckResult(
        check_id="sms_thread_slug_stable",
        passed=True,
        detail=f"Thread slug '{thread}' is stable and well-formed",
        category="sms",
    )


def check_no_sms_in_wrong_era(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that SMS tools aren't used in settings where phones don't exist.

    If the campaign setting is pre-modern (medieval, 1940s, fantasy, etc.),
    SMS tools should never be called.

    Params:
        era_has_phones: bool — whether the setting has phones (default: inferred from state)
    """
    era_has_phones = params.get("era_has_phones")

    if era_has_phones is None:
        # Try to infer from setting
        setting = state.campaign_state.get("setting", "")
        if isinstance(setting, str):
            setting_lower = setting.lower()
            no_phone_indicators = [
                "medieval", "fantasy", "wonderland", "fairy",
                "ancient", "viking", "pirate", "1800",
                "1700", "1600", "1500", "victorian",
            ]
            era_has_phones = not any(ind in setting_lower for ind in no_phone_indicators)

    if era_has_phones is False:
        # Check if an SMS tool was used
        if parsed.is_tool_call:
            tool = parsed.parsed_json.get("tool_call", "")
            if tool.startswith("sms_"):
                return CheckResult(
                    check_id="no_sms_in_wrong_era",
                    passed=False,
                    detail=f"SMS tool '{tool}' used in a setting without phones",
                    category="sms",
                )

    return CheckResult(
        check_id="no_sms_in_wrong_era",
        passed=True,
        detail="SMS usage appropriate for era",
        category="sms",
    )


def check_sms_read_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Validate sms_read tool call fields.

    Engine expects:
    - thread: required, non-empty string, ≤80 chars
    - limit (optional): integer, 1–40
    """
    if parsed.parsed_json.get("tool_call") != "sms_read":
        return CheckResult(
            check_id="sms_read_valid",
            passed=True,
            detail="Not an sms_read tool call, skipped",
            category="sms",
        )

    data = parsed.parsed_json
    issues: list[str] = []

    # thread validation (required)
    thread = data.get("thread")
    if thread is None or not isinstance(thread, str) or not thread.strip():
        issues.append("Missing or empty 'thread' field")
    elif len(thread) > SMS_FIELD_MAX_CHARS:
        issues.append(f"'thread' is {len(thread)} chars (max {SMS_FIELD_MAX_CHARS})")

    # limit validation (optional)
    limit = data.get("limit")
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int):
            issues.append(f"'limit' must be an integer, got {type(limit).__name__}")
        elif limit < SMS_READ_LIMIT_MIN or limit > SMS_READ_LIMIT_MAX:
            issues.append(
                f"'limit' {limit} out of range [{SMS_READ_LIMIT_MIN}, {SMS_READ_LIMIT_MAX}]"
            )

    if issues:
        return CheckResult(
            check_id="sms_read_valid",
            passed=False,
            detail="; ".join(issues),
            category="sms",
        )
    return CheckResult(
        check_id="sms_read_valid",
        passed=True,
        detail="sms_read fields valid",
        category="sms",
    )


def check_sms_list_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Validate sms_list tool call fields.

    Engine expects:
    - wildcard (optional): string if present
    """
    if parsed.parsed_json.get("tool_call") != "sms_list":
        return CheckResult(
            check_id="sms_list_valid",
            passed=True,
            detail="Not an sms_list tool call, skipped",
            category="sms",
        )

    data = parsed.parsed_json
    issues: list[str] = []

    # wildcard validation (optional)
    wildcard = data.get("wildcard")
    if wildcard is not None and not isinstance(wildcard, str):
        issues.append(f"'wildcard' must be a string, got {type(wildcard).__name__}")

    if issues:
        return CheckResult(
            check_id="sms_list_valid",
            passed=False,
            detail="; ".join(issues),
            category="sms",
        )
    return CheckResult(
        check_id="sms_list_valid",
        passed=True,
        detail="sms_list fields valid",
        category="sms",
    )
