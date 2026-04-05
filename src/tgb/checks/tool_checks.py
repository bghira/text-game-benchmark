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
        "memory_store": ["category", "memory"],
        "memory_terms": [],
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
        "autobiography_append": ["entries"],
        "autobiography_update": ["entries"],
        "autobiography_compress": ["character"],
        "song_search": ["query"],
    }

    # Known optional keys per tool type (not required but allowed)
    tool_optional_keys: dict[str, set[str]] = {
        "name_generate": {"origins", "gender", "context", "count"},
        "source_browse": {"document_key", "wildcard"},
        "recent_turns": {"limit"},
        "ready_to_write": {"speakers", "listeners"},
        "autobiography_append": set(),
        "autobiography_update": set(),
        "autobiography_compress": set(),
        "memory_search": {"category", "before_lines", "after_lines", "search_within", "full_text", "keep_memory_turns", "search_within_turn_ids"},
        "memory_store": {"term"},
        "memory_terms": {"wildcard"},
        "memory_turn": set(),
        "sms_read": {"limit"},
        "sms_list": {"wildcard"},
        "song_search": {"sender", "message"},
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


# ── Autobiography checks ─────────────────────────────────────

from tgb.checks.limits import (
    AUTOBIOGRAPHY_FIELD_MAX_CHARS,
    AUTOBIOGRAPHY_TRIGGER_MAX_CHARS,
    AUTOBIOGRAPHY_IMPORTANCE_MAX_CHARS,
    AUTOBIOGRAPHY_MAX_ENTRIES_PER_CALL,
)

_AUTOBIOGRAPHY_CONTENT_FIELDS = {"a", "b", "c", "text"}


def check_autobiography_append_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Validate autobiography_append / autobiography_update tool call.

    Engine expects:
    - entries: list of dicts (max 16)
    - Each entry: character (slug in state.characters),
      at least one of a/b/c/text (string, ≤600 chars),
      optional trigger (string, ≤80 chars),
      optional importance (string, ≤40 chars)
    """
    tool = parsed.parsed_json.get("tool_call", "")
    if tool not in ("autobiography_append", "autobiography_update"):
        return CheckResult(
            check_id="autobiography_append_valid",
            passed=True,
            detail="Not an autobiography_append tool call, skipped",
            category="tool_usage",
        )

    issues: list[str] = []
    entries = parsed.parsed_json.get("entries")

    if entries is None:
        issues.append("Missing 'entries' field")
    elif not isinstance(entries, list):
        issues.append(f"'entries' must be a list, got {type(entries).__name__}")
    else:
        if len(entries) > AUTOBIOGRAPHY_MAX_ENTRIES_PER_CALL:
            issues.append(
                f"Too many entries: {len(entries)} (max {AUTOBIOGRAPHY_MAX_ENTRIES_PER_CALL})"
            )
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                issues.append(f"entries[{i}] is not a dict")
                continue

            # character slug
            char_slug = entry.get("character")
            if not char_slug or not isinstance(char_slug, str):
                issues.append(f"entries[{i}] missing or invalid 'character' slug")
            elif char_slug not in state.characters:
                issues.append(f"entries[{i}] character '{char_slug}' not in known characters")

            # At least one content field
            has_content = any(
                isinstance(entry.get(f), str) and entry.get(f).strip()
                for f in _AUTOBIOGRAPHY_CONTENT_FIELDS
            )
            if not has_content:
                issues.append(f"entries[{i}] has no content (need at least one of a/b/c/text)")

            # Field length checks
            for f in _AUTOBIOGRAPHY_CONTENT_FIELDS:
                val = entry.get(f)
                if val is not None:
                    if not isinstance(val, str):
                        issues.append(f"entries[{i}].{f} must be a string")
                    elif len(val) > AUTOBIOGRAPHY_FIELD_MAX_CHARS:
                        issues.append(
                            f"entries[{i}].{f} is {len(val)} chars "
                            f"(max {AUTOBIOGRAPHY_FIELD_MAX_CHARS})"
                        )

            # trigger
            trigger = entry.get("trigger")
            if trigger is not None:
                if not isinstance(trigger, str):
                    issues.append(f"entries[{i}].trigger must be a string")
                elif len(trigger) > AUTOBIOGRAPHY_TRIGGER_MAX_CHARS:
                    issues.append(
                        f"entries[{i}].trigger is {len(trigger)} chars "
                        f"(max {AUTOBIOGRAPHY_TRIGGER_MAX_CHARS})"
                    )

            # importance
            importance = entry.get("importance")
            if importance is not None:
                if not isinstance(importance, str):
                    issues.append(f"entries[{i}].importance must be a string")
                elif len(importance) > AUTOBIOGRAPHY_IMPORTANCE_MAX_CHARS:
                    issues.append(
                        f"entries[{i}].importance is {len(importance)} chars "
                        f"(max {AUTOBIOGRAPHY_IMPORTANCE_MAX_CHARS})"
                    )

    if issues:
        detail = "; ".join(issues[:5])
        if len(issues) > 5:
            detail += f" (and {len(issues) - 5} more)"
        return CheckResult(
            check_id="autobiography_append_valid",
            passed=False,
            detail=detail,
            category="tool_usage",
        )
    return CheckResult(
        check_id="autobiography_append_valid",
        passed=True,
        detail="autobiography_append fields valid",
        category="tool_usage",
    )


def check_autobiography_compress_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Validate autobiography_compress tool call.

    Engine expects:
    - character: string slug, must exist in state.characters
    """
    if parsed.parsed_json.get("tool_call") != "autobiography_compress":
        return CheckResult(
            check_id="autobiography_compress_valid",
            passed=True,
            detail="Not an autobiography_compress tool call, skipped",
            category="tool_usage",
        )

    issues: list[str] = []
    character = parsed.parsed_json.get("character")

    if character is None or not isinstance(character, str):
        issues.append("Missing or invalid 'character' field")
    elif character not in state.characters:
        issues.append(f"character '{character}' not in known characters")

    if issues:
        return CheckResult(
            check_id="autobiography_compress_valid",
            passed=False,
            detail="; ".join(issues),
            category="tool_usage",
        )
    return CheckResult(
        check_id="autobiography_compress_valid",
        passed=True,
        detail="autobiography_compress fields valid",
        category="tool_usage",
    )


# ── Inline tool_calls array checks ─────────────────────

INLINE_TOOL_CALLS_ALLOWED = {"sms_write", "sms_schedule", "plot_plan", "chapter_plan", "song_search"}


def check_inline_tool_calls_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Validate inline tool_calls array in final JSON response.

    Engine allows an array of side-effect tool invocations alongside narration.
    Only tools in INLINE_TOOL_CALLS_ALLOWED (currently: sms_write, sms_schedule,
    plot_plan, chapter_plan, song_search) are permitted in this array.
    Each entry must be a dict with a valid "tool_call" key and correct fields.
    """
    tool_calls = parsed.parsed_json.get("tool_calls")
    if tool_calls is None:
        return CheckResult(
            check_id="inline_tool_calls_valid",
            passed=True,
            detail="No tool_calls array, skipped",
            category="tool_usage",
        )

    if not isinstance(tool_calls, list):
        return CheckResult(
            check_id="inline_tool_calls_valid",
            passed=False,
            detail=f"tool_calls must be a list, got {type(tool_calls).__name__}",
            category="tool_usage",
        )

    issues: list[str] = []
    for i, tc in enumerate(tool_calls):
        if not isinstance(tc, dict):
            issues.append(f"tool_calls[{i}] is {type(tc).__name__}, expected dict")
            continue

        tool_name = tc.get("tool_call")
        if not tool_name or not isinstance(tool_name, str):
            issues.append(f"tool_calls[{i}] missing or invalid 'tool_call' key")
            continue

        if tool_name not in INLINE_TOOL_CALLS_ALLOWED:
            issues.append(
                f"tool_calls[{i}] tool '{tool_name}' not allowed inline "
                f"(only {sorted(INLINE_TOOL_CALLS_ALLOWED)})"
            )
            continue

        # Validate required fields and reject unexpected keys per tool type
        required = _INLINE_REQUIRED.get(tool_name, [])
        missing = [k for k in required if k not in tc]
        if missing:
            issues.append(f"tool_calls[{i}] '{tool_name}' missing keys: {missing}")

        optional = _INLINE_OPTIONAL.get(tool_name, set())
        allowed = {"tool_call"} | set(required) | optional
        extra = [k for k in tc if k not in allowed]
        if extra:
            issues.append(f"tool_calls[{i}] '{tool_name}' unexpected keys: {extra}")

    if issues:
        return CheckResult(
            check_id="inline_tool_calls_valid",
            passed=False,
            detail="; ".join(issues[:5]),
            category="tool_usage",
        )
    return CheckResult(
        check_id="inline_tool_calls_valid",
        passed=True,
        detail=f"tool_calls array valid ({len(tool_calls)} entries)",
        category="tool_usage",
    )


# Required/optional keys for inline tool_calls, mirroring default_keys_map/tool_optional_keys
_INLINE_REQUIRED: dict[str, list[str]] = {
    "sms_write": ["thread", "from", "to", "message"],
    "sms_schedule": ["thread", "from", "to", "message", "delay_seconds"],
    "plot_plan": [],
    "chapter_plan": ["action"],
    "song_search": ["query"],
}

_INLINE_OPTIONAL: dict[str, set[str]] = {
    "sms_write": set(),
    "sms_schedule": set(),
    "plot_plan": {"plans"},
    "chapter_plan": {"chapter", "title", "summary", "scenes", "active", "to_scene", "resolution"},
    "song_search": {"sender", "message"},
}
