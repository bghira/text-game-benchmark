"""Mechanics checks: dice_check_valid, puzzle_trigger_valid, minigame_challenge_valid."""

from __future__ import annotations

from typing import Any

from tgb.checks.base import CheckResult
from tgb.config import Scenario, TurnSpec
from tgb.prompt_builder import AccumulatedState
from tgb.response_parser import ParsedResponse

# ── dice_check validation ──────────────────────────────────────

DICE_CHECK_REQUIRED = {"attribute", "dc", "context", "on_success", "on_failure"}
OUTCOME_EXPECTED_FIELDS = {"narration", "state_update", "player_state_update", "xp_awarded"}


def check_dice_check_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that dice_check (if present) has valid structure.

    Required: attribute (str), dc (int), context (str),
    on_success (dict with narration/state_update/player_state_update/xp_awarded),
    on_failure (same shape).
    """
    dc = parsed.parsed_json.get("dice_check")
    if dc is None:
        return CheckResult(
            check_id="dice_check_valid",
            passed=True,
            detail="No dice_check",
            category="mechanics",
        )

    if not isinstance(dc, dict):
        return CheckResult(
            check_id="dice_check_valid",
            passed=False,
            detail=f"dice_check is {type(dc).__name__}, expected dict",
            category="mechanics",
        )

    issues: list[str] = []

    # Required top-level fields
    missing = DICE_CHECK_REQUIRED - set(dc.keys())
    if missing:
        issues.append(f"missing required fields: {sorted(missing)}")

    # Type checks
    if "attribute" in dc and not isinstance(dc["attribute"], str):
        issues.append("attribute must be a string")
    if "dc" in dc:
        dc_val = dc["dc"]
        if isinstance(dc_val, bool) or not isinstance(dc_val, (int, float)):
            issues.append("dc must be an integer")
        elif isinstance(dc_val, float) and dc_val != int(dc_val):
            issues.append(f"dc must be an integer, got {dc_val}")
    if "context" in dc and not isinstance(dc["context"], str):
        issues.append("context must be a string")

    # Validate on_success and on_failure structure
    for outcome_key in ("on_success", "on_failure"):
        outcome = dc.get(outcome_key)
        if outcome is None:
            continue
        if not isinstance(outcome, dict):
            issues.append(f"{outcome_key} must be a dict")
            continue
        if "narration" not in outcome:
            issues.append(f"{outcome_key} missing 'narration'")
        elif not isinstance(outcome["narration"], str):
            issues.append(f"{outcome_key}.narration must be a string")

    if issues:
        return CheckResult(
            check_id="dice_check_valid",
            passed=False,
            detail="; ".join(issues),
            category="mechanics",
        )
    return CheckResult(
        check_id="dice_check_valid",
        passed=True,
        detail="dice_check structure valid",
        category="mechanics",
    )


# ── puzzle_trigger validation ──────────────────────────────────

PUZZLE_TYPES = {"riddle", "math", "sequence", "cipher"}
PUZZLE_DIFFICULTIES = {"easy", "medium", "hard"}
PUZZLE_REQUIRED = {"puzzle_type", "context"}


def check_puzzle_trigger_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that puzzle_trigger (if present) has valid structure.

    Required: puzzle_type (str), context (str).
    Optional: difficulty (str, one of easy/medium/hard).
    """
    pt = parsed.parsed_json.get("puzzle_trigger")
    if pt is None:
        return CheckResult(
            check_id="puzzle_trigger_valid",
            passed=True,
            detail="No puzzle_trigger",
            category="mechanics",
        )

    if not isinstance(pt, dict):
        return CheckResult(
            check_id="puzzle_trigger_valid",
            passed=False,
            detail=f"puzzle_trigger is {type(pt).__name__}, expected dict",
            category="mechanics",
        )

    issues: list[str] = []

    missing = PUZZLE_REQUIRED - set(pt.keys())
    if missing:
        issues.append(f"missing required fields: {sorted(missing)}")

    ptype = pt.get("puzzle_type")
    if isinstance(ptype, str) and ptype not in PUZZLE_TYPES:
        issues.append(f"puzzle_type '{ptype}' not in {sorted(PUZZLE_TYPES)}")

    difficulty = pt.get("difficulty")
    if difficulty is not None and isinstance(difficulty, str) and difficulty not in PUZZLE_DIFFICULTIES:
        issues.append(f"difficulty '{difficulty}' not in {sorted(PUZZLE_DIFFICULTIES)}")

    context = pt.get("context")
    if context is not None and not isinstance(context, str):
        issues.append("context must be a string")

    if issues:
        return CheckResult(
            check_id="puzzle_trigger_valid",
            passed=False,
            detail="; ".join(issues),
            category="mechanics",
        )
    return CheckResult(
        check_id="puzzle_trigger_valid",
        passed=True,
        detail=f"puzzle_trigger valid (type={pt.get('puzzle_type')})",
        category="mechanics",
    )


# ── minigame_challenge validation ──────────────────────────────

MINIGAME_TYPES = {"tic_tac_toe", "nim", "dice_duel", "coin_flip"}
MINIGAME_REQUIRED = {"game_type", "opponent_slug"}


def check_minigame_challenge_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that minigame_challenge (if present) has valid structure.

    Required: game_type (str), opponent_slug (str).
    Optional: stakes (str).
    """
    mc = parsed.parsed_json.get("minigame_challenge")
    if mc is None:
        return CheckResult(
            check_id="minigame_challenge_valid",
            passed=True,
            detail="No minigame_challenge",
            category="mechanics",
        )

    if not isinstance(mc, dict):
        return CheckResult(
            check_id="minigame_challenge_valid",
            passed=False,
            detail=f"minigame_challenge is {type(mc).__name__}, expected dict",
            category="mechanics",
        )

    issues: list[str] = []

    missing = MINIGAME_REQUIRED - set(mc.keys())
    if missing:
        issues.append(f"missing required fields: {sorted(missing)}")

    game_type = mc.get("game_type")
    if isinstance(game_type, str) and game_type not in MINIGAME_TYPES:
        issues.append(f"game_type '{game_type}' not in {sorted(MINIGAME_TYPES)}")

    opponent = mc.get("opponent_slug")
    if opponent is not None and not isinstance(opponent, str):
        issues.append("opponent_slug must be a string")

    if issues:
        return CheckResult(
            check_id="minigame_challenge_valid",
            passed=False,
            detail="; ".join(issues),
            category="mechanics",
        )
    return CheckResult(
        check_id="minigame_challenge_valid",
        passed=True,
        detail=f"minigame_challenge valid (type={mc.get('game_type')})",
        category="mechanics",
    )
