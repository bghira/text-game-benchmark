"""Scene output checks: scene_output_valid, scene_output_beats_valid."""

from __future__ import annotations

from typing import Any

from tgb.checks.base import CheckResult
from tgb.config import Scenario, TurnSpec
from tgb.prompt_builder import AccumulatedState
from tgb.response_parser import ParsedResponse

BEAT_REQUIRED_FIELDS = {"type", "text"}
BEAT_OPTIONAL_FIELDS = {
    "reasoning", "speaker", "actors", "listeners",
    "visibility", "aware_actor_ids", "aware_npc_slugs",
    "location_key", "context_key",
}
BEAT_VALID_FIELDS = BEAT_REQUIRED_FIELDS | BEAT_OPTIONAL_FIELDS

BEAT_VISIBILITY_VALUES = {"public", "local", "private", "limited"}
BEAT_TYPE_VALUES = {
    "narration", "dialogue", "player_action", "action",
    "internal", "transition", "description", "system",
}


def check_scene_output_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that scene_output (if present) has valid structure.

    The engine expects scene_output to be a dict with a 'beats' array.
    Each beat has required fields (type, text) and optional fields
    (reasoning, speaker, actors, listeners, visibility, etc.).
    """
    so = parsed.parsed_json.get("scene_output")
    if so is None:
        return CheckResult(
            check_id="scene_output_valid",
            passed=True,
            detail="No scene_output",
            category="scene_output",
        )

    if not isinstance(so, dict):
        return CheckResult(
            check_id="scene_output_valid",
            passed=False,
            detail=f"scene_output is {type(so).__name__}, expected dict",
            category="scene_output",
        )

    beats = so.get("beats")
    if beats is None:
        return CheckResult(
            check_id="scene_output_valid",
            passed=True,
            detail="scene_output present but no beats key (legacy format)",
            category="scene_output",
        )

    if not isinstance(beats, list):
        return CheckResult(
            check_id="scene_output_valid",
            passed=False,
            detail=f"scene_output.beats is {type(beats).__name__}, expected list",
            category="scene_output",
        )

    issues: list[str] = []
    for i, beat in enumerate(beats):
        if not isinstance(beat, dict):
            issues.append(f"beats[{i}] is {type(beat).__name__}, expected dict")
            continue

        # Check required fields
        for field in BEAT_REQUIRED_FIELDS:
            if field not in beat:
                issues.append(f"beats[{i}] missing required field '{field}'")

        # Check text is a string
        text = beat.get("text")
        if text is not None and not isinstance(text, str):
            issues.append(f"beats[{i}].text is {type(text).__name__}, expected str")

        # Check type value
        beat_type = beat.get("type")
        if isinstance(beat_type, str) and beat_type not in BEAT_TYPE_VALUES:
            # Allow custom types but note it
            pass

        # Check visibility value
        vis = beat.get("visibility")
        if isinstance(vis, str) and vis not in BEAT_VISIBILITY_VALUES:
            issues.append(f"beats[{i}].visibility '{vis}' not in {BEAT_VISIBILITY_VALUES}")

        # Check list fields are actually lists
        for list_field in ("actors", "listeners", "aware_actor_ids", "aware_npc_slugs"):
            val = beat.get(list_field)
            if val is not None and not isinstance(val, list):
                issues.append(f"beats[{i}].{list_field} is {type(val).__name__}, expected list")

    if issues:
        return CheckResult(
            check_id="scene_output_valid",
            passed=False,
            detail="; ".join(issues[:5]),
            category="scene_output",
        )

    return CheckResult(
        check_id="scene_output_valid",
        passed=True,
        detail=f"scene_output valid with {len(beats)} beats",
        category="scene_output",
    )
