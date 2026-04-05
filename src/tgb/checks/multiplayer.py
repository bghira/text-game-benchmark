"""Multiplayer output checks — validates co_located_player_slugs and other_player_state_updates."""

from __future__ import annotations

from typing import Any

from tgb.checks.base import CheckResult
from tgb.config import Scenario, TurnSpec
from tgb.prompt_builder import AccumulatedState, _player_slug_key
from tgb.response_parser import ParsedResponse


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


def _get_acting_player_slug(state: AccumulatedState, scenario: Scenario) -> str:
    """Get the slug of the acting player."""
    # Find the is_actor entry in party
    for entry in scenario.party:
        if entry.get("is_actor"):
            slug = entry.get("player_slug", "")
            if slug:
                return slug
            name = entry.get("character_name", entry.get("name", ""))
            if name:
                return _player_slug_key(name)
    # Fallback to player state
    player_name = state.player_state.get("character_name", "")
    if player_name:
        return _player_slug_key(player_name)
    return ""


def check_co_located_slugs_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Validate co_located_player_slugs field when present.

    Engine expects:
    - Value is a list of strings
    - Each slug exists in scenario.party
    - Acting player's slug is not included (engine auto-excludes it)
    """
    co_located = parsed.parsed_json.get("co_located_player_slugs")
    if co_located is None:
        return CheckResult(
            check_id="co_located_slugs_valid",
            passed=True,
            detail="No co_located_player_slugs field, skipped",
            category="multiplayer",
        )

    issues: list[str] = []

    if not isinstance(co_located, list):
        issues.append(f"co_located_player_slugs must be a list, got {type(co_located).__name__}")
    else:
        known_slugs = _get_party_slugs(state, scenario)
        acting_slug = _get_acting_player_slug(state, scenario)

        for i, slug in enumerate(co_located):
            if not isinstance(slug, str):
                issues.append(f"co_located_player_slugs[{i}] is not a string")
                continue
            if known_slugs and slug not in known_slugs:
                issues.append(
                    f"co_located_player_slugs[{i}] '{slug}' not in known party slugs"
                )
            if acting_slug and slug == acting_slug:
                issues.append(
                    f"co_located_player_slugs includes acting player '{slug}' (engine auto-excludes)"
                )

    if issues:
        return CheckResult(
            check_id="co_located_slugs_valid",
            passed=False,
            detail="; ".join(issues),
            category="multiplayer",
        )
    return CheckResult(
        check_id="co_located_slugs_valid",
        passed=True,
        detail="co_located_player_slugs valid",
        category="multiplayer",
    )


def check_other_player_updates_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Validate other_player_state_updates field when present.

    Engine expects:
    - Value is a dict
    - Each key is a string matching a known party slug
    - Acting player's slug is not a key (can't update yourself via this field)
    - Each value is a dict
    """
    updates = parsed.parsed_json.get("other_player_state_updates")
    if updates is None:
        return CheckResult(
            check_id="other_player_updates_valid",
            passed=True,
            detail="No other_player_state_updates field, skipped",
            category="multiplayer",
        )

    issues: list[str] = []

    if not isinstance(updates, dict):
        issues.append(
            f"other_player_state_updates must be a dict, got {type(updates).__name__}"
        )
    else:
        known_slugs = _get_party_slugs(state, scenario)
        acting_slug = _get_acting_player_slug(state, scenario)

        for key, value in updates.items():
            if not isinstance(key, str):
                issues.append(f"other_player_state_updates key {key!r} is not a string")
                continue
            if known_slugs and key not in known_slugs:
                issues.append(
                    f"other_player_state_updates key '{key}' not in known party slugs"
                )
            if acting_slug and key == acting_slug:
                issues.append(
                    f"other_player_state_updates includes acting player '{key}' "
                    "(use player_state_update instead)"
                )
            if not isinstance(value, dict):
                issues.append(
                    f"other_player_state_updates['{key}'] must be a dict, "
                    f"got {type(value).__name__}"
                )

    if issues:
        return CheckResult(
            check_id="other_player_updates_valid",
            passed=False,
            detail="; ".join(issues),
            category="multiplayer",
        )
    return CheckResult(
        check_id="other_player_updates_valid",
        passed=True,
        detail="other_player_state_updates valid",
        category="multiplayer",
    )


def check_present_characters_no_players(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that real players don't appear in character_updates as if they were NPCs.

    Engine excludes real players from present_characters. The model should not
    create or update character entries for actual party members — only NPCs
    belong in character_updates.
    """
    char_updates = parsed.parsed_json.get("character_updates")
    if not isinstance(char_updates, dict) or not char_updates:
        return CheckResult(
            check_id="present_characters_no_players",
            passed=True,
            detail="No character_updates",
            category="multiplayer",
        )

    if not scenario.party:
        return CheckResult(
            check_id="present_characters_no_players",
            passed=True,
            detail="No party defined, skipped",
            category="multiplayer",
        )

    # Build set of player slugs and names
    player_identifiers: set[str] = set()
    for entry in scenario.party:
        slug = entry.get("player_slug", "")
        if slug:
            player_identifiers.add(slug)
        name = entry.get("character_name", entry.get("name", ""))
        if name:
            player_identifiers.add(_player_slug_key(name))
            player_identifiers.add(str(name).strip().lower())
    player_identifiers.discard("")

    # Also include the acting player from player state
    player_name = state.player_state.get("character_name", "")
    if player_name:
        player_identifiers.add(_player_slug_key(player_name))
        player_identifiers.add(str(player_name).strip().lower())

    collisions: list[str] = []
    for slug in char_updates:
        if not isinstance(slug, str):
            continue
        slug_lower = slug.strip().lower()
        if slug_lower in player_identifiers or slug in player_identifiers:
            collisions.append(slug)

    if collisions:
        return CheckResult(
            check_id="present_characters_no_players",
            passed=False,
            detail=f"Real player(s) found in character_updates: {collisions} "
                   f"(players should not appear as NPCs)",
            category="multiplayer",
        )
    return CheckResult(
        check_id="present_characters_no_players",
        passed=True,
        detail="No real players in character_updates",
        category="multiplayer",
    )
