"""Privacy checks — validates turn_visibility and calendar event privacy fields.

The engine supports cross-thread privacy for multi-player campaigns:
- turn_visibility: scope (public/private/limited/local), player_slugs, npc_slugs
- calendar events: known_by, target_player(s)
- SMS/phone turns forced private; command lines stripped from narration

These checks validate structural correctness and contextual appropriateness.
"""

from __future__ import annotations

import re
from typing import Any

from tgb.checks.base import CheckResult
from tgb.checks.limits import (
    VISIBILITY_REASON_MAX_CHARS,
    CALENDAR_NAME_MAX_CHARS,
    CALENDAR_TARGET_MAX_CHARS,
)
from tgb.config import Scenario, TurnSpec
from tgb.prompt_builder import AccumulatedState, _player_slug_key
from tgb.response_parser import ParsedResponse


def _slug_valid(slug: str) -> bool:
    """Check if a string is a valid kebab-case slug."""
    return bool(re.match(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$", slug))


def _get_party_slugs(state: AccumulatedState, scenario: Scenario) -> set[str]:
    """Extract all known player slugs from party snapshot."""
    slugs: set[str] = set()
    for entry in scenario.party:
        slug = entry.get("player_slug", "")
        if slug:
            slugs.add(slug)
        name = entry.get("character_name", entry.get("name", ""))
        if name:
            slugs.add(_player_slug_key(name))
    # Always include the active player
    player_name = state.player_state.get("character_name", "")
    if player_name:
        slugs.add(_player_slug_key(player_name))
    slugs.discard("")
    return slugs


def _get_npc_slugs(state: AccumulatedState) -> set[str]:
    """Extract all known NPC slugs from world characters."""
    return set(state.characters.keys())


# ── turn_visibility checks ──────────────────────────────────────────


def check_visibility_fields_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Validate turn_visibility field structure when present.

    If the model outputs turn_visibility, it must have valid scope,
    valid player_slugs (kebab-case), and optional npc_slugs and reason.
    """
    visibility = parsed.parsed_json.get("turn_visibility")
    if visibility is None:
        # Not present — acceptable (defaults to public)
        return CheckResult(
            check_id="visibility_fields_valid",
            passed=True,
            detail="No turn_visibility field, defaults to public",
            category="privacy",
        )

    if not isinstance(visibility, dict):
        return CheckResult(
            check_id="visibility_fields_valid",
            passed=False,
            detail=f"turn_visibility must be an object, got {type(visibility).__name__}",
            category="privacy",
        )

    issues = []

    # Scope validation
    scope = visibility.get("scope", "")
    if not scope:
        issues.append("missing 'scope'")
    elif scope not in ("public", "private", "limited", "local"):
        issues.append(f"scope '{scope}' not one of public/private/limited/local")

    # player_slugs validation
    player_slugs = visibility.get("player_slugs")
    if player_slugs is not None:
        if not isinstance(player_slugs, list):
            issues.append("player_slugs must be an array")
        else:
            for i, slug in enumerate(player_slugs):
                if not isinstance(slug, str):
                    issues.append(f"player_slugs[{i}] not a string")
                elif not _slug_valid(slug):
                    issues.append(f"player_slugs[{i}] '{slug}' not valid kebab-case")

    # npc_slugs validation
    npc_slugs = visibility.get("npc_slugs")
    if npc_slugs is not None:
        if not isinstance(npc_slugs, list):
            issues.append("npc_slugs must be an array")
        else:
            for i, slug in enumerate(npc_slugs):
                if not isinstance(slug, str):
                    issues.append(f"npc_slugs[{i}] not a string")
                elif not _slug_valid(slug):
                    issues.append(f"npc_slugs[{i}] '{slug}' not valid kebab-case")

    # reason validation
    reason = visibility.get("reason")
    if reason is not None:
        if not isinstance(reason, str):
            issues.append("reason must be a string")
        elif len(reason) > VISIBILITY_REASON_MAX_CHARS:
            issues.append(f"reason exceeds {VISIBILITY_REASON_MAX_CHARS} chars ({len(reason)})")

    # limited scope requires player_slugs
    if scope == "limited":
        if not player_slugs or not isinstance(player_slugs, list) or len(player_slugs) == 0:
            issues.append("'limited' scope requires non-empty player_slugs")

    if issues:
        return CheckResult(
            check_id="visibility_fields_valid",
            passed=False,
            detail="; ".join(issues),
            category="privacy",
        )
    return CheckResult(
        check_id="visibility_fields_valid",
        passed=True,
        detail=f"Valid turn_visibility (scope={scope})",
        category="privacy",
    )


def check_visibility_scope_present(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that multi-player turns include turn_visibility.

    When the scenario is multi-player and TURN_VISIBILITY_DEFAULT is set,
    the model should output turn_visibility on every turn.
    """
    expect_visibility = params.get("expect_visibility", False)
    if not expect_visibility and not scenario.campaign.multi_player:
        return CheckResult(
            check_id="visibility_scope_present",
            passed=True,
            detail="Not a multi-player scenario, skipped",
            category="privacy",
        )

    visibility = parsed.parsed_json.get("turn_visibility")
    if visibility is None or not isinstance(visibility, dict):
        return CheckResult(
            check_id="visibility_scope_present",
            passed=False,
            detail="Multi-player turn missing turn_visibility field",
            category="privacy",
        )
    scope = visibility.get("scope", "")
    if scope not in ("public", "private", "limited", "local"):
        return CheckResult(
            check_id="visibility_scope_present",
            passed=False,
            detail=f"turn_visibility.scope '{scope}' not valid",
            category="privacy",
        )
    return CheckResult(
        check_id="visibility_scope_present",
        passed=True,
        detail=f"turn_visibility present with scope={scope}",
        category="privacy",
    )


def check_visibility_default_respected(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that the model respects TURN_VISIBILITY_DEFAULT.

    Rules:
    - If default is 'private', scope must be private or limited (not public/local).
    - If default is 'local', scope should be local/private/limited (not public).
    - If default is 'public', any scope is fine.
    """
    effective_default = turn.turn_visibility_default

    if effective_default == "public":
        return CheckResult(
            check_id="visibility_default_respected",
            passed=True,
            detail="Default is public, no restriction to check",
            category="privacy",
        )

    visibility = parsed.parsed_json.get("turn_visibility")
    if visibility is None:
        # No visibility = implicit public, but default is restrictive
        return CheckResult(
            check_id="visibility_default_respected",
            passed=False,
            detail=f"Default is {effective_default} but no turn_visibility emitted (implicit public)",
            category="privacy",
        )

    if not isinstance(visibility, dict):
        return CheckResult(
            check_id="visibility_default_respected",
            passed=False,
            detail="turn_visibility is not a dict",
            category="privacy",
        )

    scope = str(visibility.get("scope", "")).strip().lower()

    if effective_default == "private":
        # Private default: only private/limited are acceptable
        if scope in ("private", "limited"):
            return CheckResult(
                check_id="visibility_default_respected",
                passed=True,
                detail=f"Default is private and scope is {scope}",
                category="privacy",
            )
        return CheckResult(
            check_id="visibility_default_respected",
            passed=False,
            detail=f"Default is private but model set scope to {scope}",
            category="privacy",
        )

    if effective_default == "local":
        # Local default: local/private/limited are acceptable, public is not
        if scope in ("local", "private", "limited"):
            return CheckResult(
                check_id="visibility_default_respected",
                passed=True,
                detail=f"Default is local and scope is {scope}",
                category="privacy",
            )
        return CheckResult(
            check_id="visibility_default_respected",
            passed=False,
            detail=f"Default is local but model set scope to {scope} (should be local/private/limited)",
            category="privacy",
        )

    # Unknown default — pass
    return CheckResult(
        check_id="visibility_default_respected",
        passed=True,
        detail=f"Default is {effective_default}, no specific restriction",
        category="privacy",
    )


def check_visibility_player_slugs_known(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that player_slugs in turn_visibility reference known players.

    Player slugs should match entries from PARTY_SNAPSHOT or scenario party.
    """
    visibility = parsed.parsed_json.get("turn_visibility")
    if not isinstance(visibility, dict):
        return CheckResult(
            check_id="visibility_player_slugs_known",
            passed=True,
            detail="No turn_visibility, skipped",
            category="privacy",
        )

    player_slugs = visibility.get("player_slugs")
    if not isinstance(player_slugs, list) or not player_slugs:
        return CheckResult(
            check_id="visibility_player_slugs_known",
            passed=True,
            detail="No player_slugs to check",
            category="privacy",
        )

    known = _get_party_slugs(state, scenario)
    if not known:
        # No party data to cross-reference — skip
        return CheckResult(
            check_id="visibility_player_slugs_known",
            passed=True,
            detail="No party data for cross-reference",
            category="privacy",
        )

    unknown = []
    for slug in player_slugs:
        if isinstance(slug, str) and slug not in known:
            unknown.append(slug)

    if unknown:
        return CheckResult(
            check_id="visibility_player_slugs_known",
            passed=False,
            detail=f"Unknown player slugs: {unknown} (known: {sorted(known)})",
            category="privacy",
        )
    return CheckResult(
        check_id="visibility_player_slugs_known",
        passed=True,
        detail=f"All {len(player_slugs)} player slugs are known",
        category="privacy",
    )


def check_visibility_npc_slugs_known(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that npc_slugs in turn_visibility reference known NPCs.

    NPC slugs should match entries from WORLD_CHARACTERS.
    """
    visibility = parsed.parsed_json.get("turn_visibility")
    if not isinstance(visibility, dict):
        return CheckResult(
            check_id="visibility_npc_slugs_known",
            passed=True,
            detail="No turn_visibility, skipped",
            category="privacy",
        )

    npc_slugs = visibility.get("npc_slugs")
    if not isinstance(npc_slugs, list) or not npc_slugs:
        return CheckResult(
            check_id="visibility_npc_slugs_known",
            passed=True,
            detail="No npc_slugs to check",
            category="privacy",
        )

    known = _get_npc_slugs(state)
    if not known:
        return CheckResult(
            check_id="visibility_npc_slugs_known",
            passed=True,
            detail="No NPC data for cross-reference",
            category="privacy",
        )

    unknown = []
    for slug in npc_slugs:
        if isinstance(slug, str) and slug not in known:
            unknown.append(slug)

    if unknown:
        return CheckResult(
            check_id="visibility_npc_slugs_known",
            passed=False,
            detail=f"Unknown NPC slugs: {unknown} (known: {sorted(known)})",
            category="privacy",
        )
    return CheckResult(
        check_id="visibility_npc_slugs_known",
        passed=True,
        detail=f"All {len(npc_slugs)} NPC slugs are known",
        category="privacy",
    )


def check_visibility_no_narration_leak(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that private/limited turns don't reference excluded players.

    If a turn is private or limited to specific players, the narration
    should not address or describe actions of players who aren't in the
    visibility list. This catches the model contradicting its own privacy
    declaration.
    """
    visibility = parsed.parsed_json.get("turn_visibility")
    if not isinstance(visibility, dict):
        return CheckResult(
            check_id="visibility_no_narration_leak",
            passed=True,
            detail="No turn_visibility, skipped",
            category="privacy",
        )

    scope = str(visibility.get("scope", "")).strip().lower()
    if scope in ("public", "local"):
        # Public/local scopes don't restrict by player — no leak possible
        return CheckResult(
            check_id="visibility_no_narration_leak",
            passed=True,
            detail=f"{scope.title()} scope, no per-player leak check needed",
            category="privacy",
        )

    narration = parsed.parsed_json.get("narration", "")
    if not isinstance(narration, str) or not narration.strip():
        return CheckResult(
            check_id="visibility_no_narration_leak",
            passed=True,
            detail="No narration to check",
            category="privacy",
        )

    visible_slugs = set()
    raw_slugs = visibility.get("player_slugs", [])
    if isinstance(raw_slugs, list):
        for s in raw_slugs:
            if isinstance(s, str):
                visible_slugs.add(s)

    # Get all party member names and check if excluded ones appear in narration
    all_party = _get_party_slugs(state, scenario)
    excluded = all_party - visible_slugs
    if not excluded:
        return CheckResult(
            check_id="visibility_no_narration_leak",
            passed=True,
            detail="No excluded players to check against",
            category="privacy",
        )

    # Look up actual names for excluded slugs from party data
    narration_lower = narration.lower()
    leaked = []
    for entry in scenario.party:
        name = entry.get("character_name", entry.get("name", ""))
        slug = entry.get("player_slug", _player_slug_key(name)) if name else ""
        if slug in excluded and name:
            # Use word-boundary matching to avoid false positives
            # (e.g. "Jack" matching "Hijack")
            if re.search(r"\b" + re.escape(name.lower()) + r"\b", narration_lower):
                leaked.append(name)

    if leaked:
        return CheckResult(
            check_id="visibility_no_narration_leak",
            passed=False,
            detail=f"Private/limited turn narration mentions excluded player(s): {leaked}",
            category="privacy",
        )
    return CheckResult(
        check_id="visibility_no_narration_leak",
        passed=True,
        detail=f"No excluded player names found in {scope} narration",
        category="privacy",
    )


# ── Calendar privacy checks ─────────────────────────────────────────


def check_calendar_known_by_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Validate known_by field on calendar event additions.

    When a model adds calendar events with known_by, the names should
    reference known characters from WORLD_CHARACTERS or PARTY_SNAPSHOT.
    """
    calendar_update = parsed.parsed_json.get("calendar_update")
    if not isinstance(calendar_update, dict):
        return CheckResult(
            check_id="calendar_known_by_valid",
            passed=True,
            detail="No calendar_update, skipped",
            category="privacy",
        )

    adds = calendar_update.get("add", [])
    if not isinstance(adds, list):
        return CheckResult(
            check_id="calendar_known_by_valid",
            passed=True,
            detail="No calendar adds, skipped",
            category="privacy",
        )

    known_npcs = _get_npc_slugs(state)
    known_party = _get_party_slugs(state, scenario)
    # Build a set of known names (lowercase) from characters + party
    known_names: set[str] = set()
    for slug in known_npcs:
        char = state.characters.get(slug, {})
        name = char.get("name", "")
        if name:
            known_names.add(name.lower())
        known_names.add(slug.lower())
    for entry in scenario.party:
        name = entry.get("character_name", entry.get("name", ""))
        if name:
            known_names.add(name.lower())
    player_name = state.player_state.get("character_name", "")
    if player_name:
        known_names.add(player_name.lower())

    issues = []
    for i, event in enumerate(adds):
        if not isinstance(event, dict):
            continue
        known_by = event.get("known_by")
        if not isinstance(known_by, list):
            continue
        for j, name in enumerate(known_by):
            if not isinstance(name, str) or not name.strip():
                issues.append(f"add[{i}].known_by[{j}] empty or not a string")
            elif len(name) > CALENDAR_NAME_MAX_CHARS:
                issues.append(f"add[{i}].known_by[{j}] exceeds {CALENDAR_NAME_MAX_CHARS} chars")
            elif known_names and name.lower() not in known_names:
                issues.append(f"add[{i}].known_by '{name}' not a known character")

    if issues:
        return CheckResult(
            check_id="calendar_known_by_valid",
            passed=False,
            detail="; ".join(issues),
            category="privacy",
        )
    return CheckResult(
        check_id="calendar_known_by_valid",
        passed=True,
        detail="Calendar known_by fields valid",
        category="privacy",
    )


def check_calendar_target_player_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Validate target_player(s) field on calendar event additions.

    When a model adds calendar events targeting specific players, the
    targets should reference known players from PARTY_SNAPSHOT.
    """
    calendar_update = parsed.parsed_json.get("calendar_update")
    if not isinstance(calendar_update, dict):
        return CheckResult(
            check_id="calendar_target_player_valid",
            passed=True,
            detail="No calendar_update, skipped",
            category="privacy",
        )

    adds = calendar_update.get("add", [])
    if not isinstance(adds, list):
        return CheckResult(
            check_id="calendar_target_player_valid",
            passed=True,
            detail="No calendar adds, skipped",
            category="privacy",
        )

    known_party = _get_party_slugs(state, scenario)
    # Also collect raw names and IDs for fuzzy matching
    known_refs: set[str] = set()
    for entry in scenario.party:
        for key in ("character_name", "name", "actor_id", "discord_mention", "player_slug"):
            val = entry.get(key, "")
            if val:
                known_refs.add(str(val).strip().lower())
    player_name = state.player_state.get("character_name", "")
    if player_name:
        known_refs.add(player_name.lower())
    known_refs |= {s.lower() for s in known_party}
    known_refs.discard("")

    issues = []
    for i, event in enumerate(adds):
        if not isinstance(event, dict):
            continue
        # Check all the target variants the engine accepts
        for key in ("target_player", "target_players"):
            targets = event.get(key)
            if targets is None:
                continue
            if not isinstance(targets, list):
                targets = [targets]
            for j, target in enumerate(targets):
                target_str = str(target or "").strip()
                if not target_str:
                    issues.append(f"add[{i}].{key}[{j}] is empty")
                elif len(target_str) > CALENDAR_TARGET_MAX_CHARS:
                    issues.append(f"add[{i}].{key}[{j}] exceeds {CALENDAR_TARGET_MAX_CHARS} chars")
                elif known_refs and target_str.lower() not in known_refs:
                    issues.append(f"add[{i}].{key} '{target_str}' not a known player")

    if issues:
        return CheckResult(
            check_id="calendar_target_player_valid",
            passed=False,
            detail="; ".join(issues),
            category="privacy",
        )
    return CheckResult(
        check_id="calendar_target_player_valid",
        passed=True,
        detail="Calendar target_player(s) valid",
        category="privacy",
    )


def check_no_public_leak_in_private_turn(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that non-public turns don't update the shared campaign summary.

    The engine silently drops summary_update for non-public turns (private,
    limited, AND local). If the model writes a meaningful summary_update AND
    marks the turn as non-public, it's confused about what's shared — the
    summary_update will be lost.
    """
    visibility = parsed.parsed_json.get("turn_visibility")
    if not isinstance(visibility, dict):
        return CheckResult(
            check_id="no_public_leak_in_private_turn",
            passed=True,
            detail="No turn_visibility, skipped",
            category="privacy",
        )

    scope = str(visibility.get("scope", "")).strip().lower()
    if scope == "public":
        return CheckResult(
            check_id="no_public_leak_in_private_turn",
            passed=True,
            detail="Public scope, summary_update is fine",
            category="privacy",
        )

    summary_update = parsed.parsed_json.get("summary_update", "")
    if isinstance(summary_update, str) and summary_update.strip():
        return CheckResult(
            check_id="no_public_leak_in_private_turn",
            passed=False,
            detail=f"Turn is {scope} but includes summary_update ('{summary_update[:60]}...') which the engine will discard",
            category="privacy",
        )
    return CheckResult(
        check_id="no_public_leak_in_private_turn",
        passed=True,
        detail=f"Turn is {scope} with no summary_update — consistent",
        category="privacy",
    )


# ── SMS privacy checks ────────────────────────────────────────────

_SMS_COMMAND_RE = re.compile(
    r"(?:i\s+)?(?:send\s+)?(?:sms|text|message)\b",
    flags=re.IGNORECASE,
)


def check_sms_not_in_narration(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that SMS command lines don't appear in narration text.

    The engine strips literal player command lines like 'I text X ...' from
    stored turns. The model should not repeat the raw command in narration
    either — the SMS log is the canonical record.
    """
    narration = parsed.parsed_json.get("narration", "")
    if not isinstance(narration, str) or not narration.strip():
        return CheckResult(
            check_id="sms_not_in_narration",
            passed=True,
            detail="No narration to check",
            category="privacy",
        )

    leaked_lines = []
    for line in narration.splitlines():
        stripped = " ".join(line.strip().split())
        if stripped and _SMS_COMMAND_RE.match(stripped):
            leaked_lines.append(stripped[:80])

    if leaked_lines:
        return CheckResult(
            check_id="sms_not_in_narration",
            passed=False,
            detail=f"SMS command line(s) echoed in narration: {leaked_lines}",
            category="privacy",
        )
    return CheckResult(
        check_id="sms_not_in_narration",
        passed=True,
        detail="No SMS command lines in narration",
        category="privacy",
    )


def check_sms_turn_private(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that SMS/phone turns use private or limited visibility.

    The engine forces phone activity to private scope. If the model outputs
    turn_visibility for a turn whose action involves texting/calling, the
    scope should be private or limited — never public or local.
    """
    action_lower = turn.action.lower()
    is_phone_action = bool(re.search(
        r"\b(?:text|sms|message|send\s+text|send\s+sms|i\s+text|i\s+message|phone|call)\b",
        action_lower,
    ))

    if not is_phone_action:
        return CheckResult(
            check_id="sms_turn_private",
            passed=True,
            detail="Not a phone/SMS action",
            category="privacy",
        )

    visibility = parsed.parsed_json.get("turn_visibility")
    if not isinstance(visibility, dict):
        # No visibility emitted — engine will auto-set to private, so acceptable
        return CheckResult(
            check_id="sms_turn_private",
            passed=True,
            detail="Phone action with no turn_visibility (engine auto-privates)",
            category="privacy",
        )

    scope = str(visibility.get("scope", "")).strip().lower()
    if scope in ("private", "limited"):
        return CheckResult(
            check_id="sms_turn_private",
            passed=True,
            detail=f"Phone/SMS turn correctly uses scope={scope}",
            category="privacy",
        )

    return CheckResult(
        check_id="sms_turn_private",
        passed=False,
        detail=f"Phone/SMS turn uses scope={scope} but should be private or limited",
        category="privacy",
    )
