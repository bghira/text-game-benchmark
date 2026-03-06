"""Agency checks: consent_respected, player_agency_respected."""

from __future__ import annotations

import re
from typing import Any

from tgb.checks.base import CheckResult
from tgb.config import Scenario, TurnSpec
from tgb.prompt_builder import AccumulatedState
from tgb.response_parser import ParsedResponse

# Patterns indicating dialogue written for another player character
OTHER_PC_DIALOGUE_PATTERNS = [
    r'"[^"]{5,}"',  # Quoted dialogue (>5 chars, could be NPC or PC)
]

# Patterns indicating re-pitching a refused offer
PRESSURE_PATTERNS = [
    r"(?i)\bwhat if\b",
    r"(?i)\bare you sure\b",
    r"(?i)\breconsider\b",
    r"(?i)\bone more chance\b",
    r"(?i)\blast chance\b",
    r"(?i)\bfinal offer\b",
    r"(?i)\bthink about it\b",
    r"(?i)\bchange your mind\b",
]


def check_consent_respected(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that a prior refusal is not re-pitched.

    Params:
        refused_topic: str — what was refused (for context matching)
    """
    narration = parsed.parsed_json.get("narration", "")
    if not isinstance(narration, str):
        return CheckResult(
            check_id="consent_respected",
            passed=True,
            detail="No narration",
            category="agency",
        )

    refused_topic = params.get("refused_topic", "")

    # Check for pressure patterns
    pressure_found = []
    for pattern in PRESSURE_PATTERNS:
        if re.search(pattern, narration):
            pressure_found.append(pattern.split(r"\b")[1] if r"\b" in pattern else pattern)

    # If a refused topic is specified, also check if the topic itself is re-mentioned
    # in a way that suggests re-pitching
    topic_repitch = False
    if refused_topic:
        if refused_topic.lower() in narration.lower():
            # Topic mentioned — check if it's in a pressure context
            for pattern in PRESSURE_PATTERNS:
                if re.search(pattern, narration):
                    topic_repitch = True
                    break

    if pressure_found and (topic_repitch or not refused_topic):
        return CheckResult(
            check_id="consent_respected",
            passed=False,
            detail=f"Pressure language after refusal: {pressure_found[:3]}",
            category="agency",
        )
    return CheckResult(
        check_id="consent_respected",
        passed=True,
        detail="No re-pitching of refused offers",
        category="agency",
    )


def check_player_agency_respected(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that no dialogue or actions are written for other player characters.

    This checks narration for signs that another PC's dialogue/actions were authored.
    """
    narration = parsed.parsed_json.get("narration", "")
    if not isinstance(narration, str):
        return CheckResult(
            check_id="player_agency_respected",
            passed=True,
            detail="No narration",
            category="agency",
        )

    # Get other player character names/mentions from party
    other_pcs: list[str] = []
    acting_user_id = scenario.player.user_id
    for member in scenario.party:
        member_mention = member.get("discord_mention", "")
        member_name = member.get("character_name", "")
        # Skip the acting player
        if str(acting_user_id) in member_mention:
            continue
        if member_name:
            other_pcs.append(member_name)

    if not other_pcs:
        return CheckResult(
            check_id="player_agency_respected",
            passed=True,
            detail="No other PCs in party",
            category="agency",
        )

    # Check if other PCs are given dialogue or actions
    violations = []
    for pc_name in other_pcs:
        # Pattern: "PCName says/said/asks..." or "PCName draws/follows/moves..."
        action_pattern = rf"(?i)\b{re.escape(pc_name)}\b\s+(says?|said|asks?|asked|draws?|drew|follows?|followed|moves?|moved|decides?|decided|grabs?|grabbed|runs?|ran)"
        if re.search(action_pattern, narration):
            violations.append(f"{pc_name} given actions/dialogue")

    if violations:
        return CheckResult(
            check_id="player_agency_respected",
            passed=False,
            detail=f"Other PC agency violations: {violations}",
            category="agency",
        )
    return CheckResult(
        check_id="player_agency_respected",
        passed=True,
        detail="Other PCs not given dialogue/actions",
        category="agency",
    )
