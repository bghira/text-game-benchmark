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
        "communication_rules": ["keys"],
        "story_outline": ["chapter"],
    }

    # Known optional keys per tool type (not required but allowed)
    tool_optional_keys: dict[str, set[str]] = {
        "name_generate": {"origins", "gender", "context", "count"},
        "source_browse": {"document_key", "wildcard"},
        "recent_turns": {"limit"},
        "ready_to_write": {"speakers", "listeners"},
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
    global_optional = {"category", "limit"}
    per_tool_optional = tool_optional_keys.get(tool_name, set())
    allowed = {"tool_call"} | global_optional | per_tool_optional | set(required_keys)
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


# ── Engine communication rule keys (mirrors ZorkEmulator.COMMUNICATION_RULE_KEYS) ──
VALID_COMMUNICATION_RULE_KEYS = {
    "GM-RULE-COMMUNICATION-SOFTENING",
    "GM-RULE-COMMUNICATION-REFRAMING",
    "GM-RULE-COMMUNICATION-TESTING",
    "GM-RULE-COMMUNICATION-PROCESSING",
    "GM-RULE-COMMUNICATION-INDIRECTION",
    "GM-RULE-COMMUNICATION-PLAYFUL",
    "GM-RULE-COMMUNICATION-ACTION",
    "GM-RULE-SUBSTANCE-EXTRACTION",
    "GM-RULE-NPC-RESPONSE-TO-INDIRECTION",
    "GM-RULE-EVASION-DEFINITION",
}

COMMUNICATION_RULES_MAX_KEYS = 8


def check_communication_rules_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Validate communication_rules tool call fields.

    Engine expects:
    - keys: string or list of strings (max 8), each uppercased and matched
      against COMMUNICATION_RULE_KEYS
    """
    if parsed.parsed_json.get("tool_call") != "communication_rules":
        return CheckResult(
            check_id="communication_rules_valid",
            passed=True,
            detail="Not a communication_rules tool call, skipped",
            category="tool_usage",
        )

    issues: list[str] = []
    keys_raw = parsed.parsed_json.get("keys")

    if keys_raw is None:
        issues.append("Missing 'keys' field")
    else:
        # Normalize to list
        if isinstance(keys_raw, str):
            keys_list = [keys_raw]
        elif isinstance(keys_raw, list):
            keys_list = keys_raw
        else:
            issues.append(f"'keys' must be a string or list, got {type(keys_raw).__name__}")
            keys_list = []

        if len(keys_list) > COMMUNICATION_RULES_MAX_KEYS:
            issues.append(f"Too many keys: {len(keys_list)} (max {COMMUNICATION_RULES_MAX_KEYS})")

        for i, key in enumerate(keys_list):
            if not isinstance(key, str):
                issues.append(f"keys[{i}] is not a string")
                continue
            normalized = key.strip().upper()
            if normalized not in VALID_COMMUNICATION_RULE_KEYS:
                issues.append(f"keys[{i}] '{key}' is not a valid communication rule key")

    if issues:
        return CheckResult(
            check_id="communication_rules_valid",
            passed=False,
            detail="; ".join(issues),
            category="tool_usage",
        )
    return CheckResult(
        check_id="communication_rules_valid",
        passed=True,
        detail="communication_rules fields valid",
        category="tool_usage",
    )


NAME_GENERATE_MAX_ORIGINS = 4
NAME_GENERATE_MAX_COUNT = 6
NAME_GENERATE_VALID_GENDERS = {"both", "m", "f", "male", "female"}
NAME_GENERATE_CONTEXT_MAX_CHARS = 300


def check_name_generate_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Validate name_generate tool call fields.

    Engine expects:
    - origins: optional array of strings (up to 4 entries)
    - gender: optional string (default "both", also "m", "f")
    - count: optional integer 1-6 (default 5)
    - context: optional string describing character concept (max 300 chars)
    """
    if parsed.parsed_json.get("tool_call") != "name_generate":
        return CheckResult(
            check_id="name_generate_valid",
            passed=True,
            detail="Not a name_generate tool call, skipped",
            category="tool_usage",
        )

    data = parsed.parsed_json
    issues: list[str] = []

    # origins validation
    origins = data.get("origins")
    if origins is not None:
        if isinstance(origins, str):
            origins = [origins]
        if isinstance(origins, list):
            if len(origins) > NAME_GENERATE_MAX_ORIGINS:
                issues.append(f"Too many origins: {len(origins)} (max {NAME_GENERATE_MAX_ORIGINS})")
            for i, o in enumerate(origins):
                if not isinstance(o, str):
                    issues.append(f"origins[{i}] is not a string")
        else:
            issues.append(f"'origins' must be a string or list, got {type(origins).__name__}")

    # gender validation
    gender = data.get("gender")
    if gender is not None:
        if not isinstance(gender, str):
            issues.append(f"'gender' must be a string, got {type(gender).__name__}")
        elif gender.strip().lower() not in NAME_GENERATE_VALID_GENDERS:
            issues.append(f"'gender' value '{gender}' not valid (expected: both, m, f)")

    # count validation
    count = data.get("count")
    if count is not None:
        if isinstance(count, bool) or not isinstance(count, (int, float)):
            issues.append(f"'count' must be an integer, got {type(count).__name__}")
        elif int(count) < 1 or int(count) > NAME_GENERATE_MAX_COUNT:
            issues.append(f"'count' {int(count)} out of range [1, {NAME_GENERATE_MAX_COUNT}]")

    # context validation
    context = data.get("context")
    if context is not None:
        if not isinstance(context, str):
            issues.append(f"'context' must be a string, got {type(context).__name__}")
        elif len(context) > NAME_GENERATE_CONTEXT_MAX_CHARS:
            issues.append(f"'context' is {len(context)} chars (max {NAME_GENERATE_CONTEXT_MAX_CHARS})")

    if issues:
        return CheckResult(
            check_id="name_generate_valid",
            passed=False,
            detail="; ".join(issues),
            category="tool_usage",
        )
    return CheckResult(
        check_id="name_generate_valid",
        passed=True,
        detail="name_generate fields valid",
        category="tool_usage",
    )


# ── ready_to_write LCD filtering checks ─────────────────────


def check_ready_to_write_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Validate ready_to_write tool call speakers/listeners fields.

    Engine expects speakers and listeners to be arrays of strings,
    each referencing a known NPC slug or player slug from the party.
    """
    if parsed.parsed_json.get("tool_call") != "ready_to_write":
        return CheckResult(
            check_id="ready_to_write_valid",
            passed=True,
            detail="Not a ready_to_write tool call, skipped",
            category="tool_usage",
        )

    data = parsed.parsed_json
    issues: list[str] = []

    # Collect known slugs: NPC slugs + player slugs from party
    known_slugs: set[str] = set(state.characters.keys())
    for entry in (scenario.party or []):
        ps = entry.get("player_slug", "")
        if ps:
            known_slugs.add(ps)
    # Single-player fallback: derive player slug from character_name
    if not scenario.party:
        char_name = state.player_state.get("character_name", "")
        if char_name:
            slug = str(char_name).strip().lower()
            import re as _re
            slug = _re.sub(r"[^a-z0-9]+", "-", slug).strip("-")[:64]
            if slug:
                known_slugs.add(slug)

    for field_name in ("speakers", "listeners"):
        val = data.get(field_name)
        if val is None:
            continue
        if not isinstance(val, list):
            issues.append(f"'{field_name}' must be an array, got {type(val).__name__}")
            continue
        for i, item in enumerate(val):
            if not isinstance(item, str):
                issues.append(f"{field_name}[{i}] is not a string")
            elif item not in known_slugs:
                issues.append(f"{field_name}[{i}] '{item}' is not a known slug")

    if issues:
        return CheckResult(
            check_id="ready_to_write_valid",
            passed=False,
            detail="; ".join(issues),
            category="tool_usage",
        )
    return CheckResult(
        check_id="ready_to_write_valid",
        passed=True,
        detail="ready_to_write fields valid",
        category="tool_usage",
    )


def check_ready_to_write_lcd_complete(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that ready_to_write includes all non-deceased NPCs at the player's location.

    In multi-player scenarios, the engine uses speakers/listeners to filter
    LCD (Least Common Denominator) context. Missing an NPC who is present
    at the player's location causes the engine to over-filter, dropping
    context that the NPC should be aware of.

    Skips for single-player scenarios (no party or party of 1) unless
    force_check param is set.
    """
    if parsed.parsed_json.get("tool_call") != "ready_to_write":
        return CheckResult(
            check_id="ready_to_write_lcd_complete",
            passed=True,
            detail="Not a ready_to_write tool call, skipped",
            category="tool_usage",
        )

    # Skip single-player unless forced
    force = params.get("force_check", False)
    is_single_player = not scenario.party or len(scenario.party) <= 1
    if not force and is_single_player:
        return CheckResult(
            check_id="ready_to_write_lcd_complete",
            passed=True,
            detail="Single-player scenario, LCD check skipped",
            category="tool_usage",
        )

    data = parsed.parsed_json
    player_location = str(state.player_state.get("location", "")).strip().lower()
    if not player_location:
        return CheckResult(
            check_id="ready_to_write_lcd_complete",
            passed=True,
            detail="No player location set, skipped",
            category="tool_usage",
        )

    # Find all non-deceased NPCs at the player's location
    npcs_at_location: set[str] = set()
    for slug, char in state.characters.items():
        if not isinstance(char, dict):
            continue
        if char.get("deceased_reason"):
            continue
        char_loc = str(char.get("location", "")).strip().lower()
        if char_loc == player_location:
            npcs_at_location.add(slug)

    if not npcs_at_location:
        return CheckResult(
            check_id="ready_to_write_lcd_complete",
            passed=True,
            detail="No NPCs at player location",
            category="tool_usage",
        )

    # Collect all slugs referenced in speakers + listeners
    referenced: set[str] = set()
    for field_name in ("speakers", "listeners"):
        val = data.get(field_name)
        if isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    referenced.add(item)

    missing = npcs_at_location - referenced
    if missing:
        return CheckResult(
            check_id="ready_to_write_lcd_complete",
            passed=False,
            detail=f"NPCs at location not in speakers/listeners: {sorted(missing)}",
            category="tool_usage",
        )

    return CheckResult(
        check_id="ready_to_write_lcd_complete",
        passed=True,
        detail=f"All {len(npcs_at_location)} NPCs at location included",
        category="tool_usage",
    )
