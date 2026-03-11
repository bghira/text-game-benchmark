"""Tests for scene_output, dice_check, puzzle_trigger, minigame_challenge validation."""

import json

from tgb.checks.scene_output import check_scene_output_valid
from tgb.checks.mechanics import (
    check_dice_check_valid,
    check_puzzle_trigger_valid,
    check_minigame_challenge_valid,
)
from tgb.checks.state_mgmt import (
    check_game_time_advanced,
    check_state_no_character_removal,
    check_inventory_changes_limit,
)
from tgb.response_parser import ParsedResponse
from tgb.config import Scenario, CampaignSetup, PlayerSetup, TurnSpec
from tgb.prompt_builder import AccumulatedState


def _make_scenario(**kwargs) -> Scenario:
    return Scenario(
        name=kwargs.get("name", "test"),
        description="test",
        tags=[],
        tier="basic",
        campaign=kwargs.get("campaign", CampaignSetup(name="test")),
        player=kwargs.get("player", PlayerSetup()),
        turns=[TurnSpec(action="test")],
    )


def _make_state(scenario=None, **overrides) -> AccumulatedState:
    scenario = scenario or _make_scenario()
    state = AccumulatedState(scenario)
    for k, v in overrides.items():
        setattr(state, k, v)
    return state


def _make_parsed(json_data=None, raw="", parse_error="", is_tool_call=False) -> ParsedResponse:
    if json_data is None:
        json_data = {}
    if not raw and json_data:
        raw = json.dumps(json_data)
    return ParsedResponse(
        raw=raw,
        parsed_json=json_data,
        parse_error=parse_error,
        is_tool_call=is_tool_call,
    )


TURN = TurnSpec(action="test")
P = {}


# ── scene_output validation ─────────────────────────────────────

class TestSceneOutputValid:
    def test_no_scene_output(self):
        parsed = _make_parsed({"narration": "hello"})
        result = check_scene_output_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_valid_scene_output_with_beats(self):
        parsed = _make_parsed({"scene_output": {
            "beats": [
                {"type": "narration", "text": "The door opens.", "speaker": "narrator"},
                {"type": "dialogue", "text": "Hello!", "speaker": "guard", "actors": ["guard"]},
            ],
        }})
        result = check_scene_output_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed
        assert "2 beats" in result.detail

    def test_scene_output_not_dict(self):
        parsed = _make_parsed({"scene_output": "bad"})
        result = check_scene_output_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed

    def test_beats_not_list(self):
        parsed = _make_parsed({"scene_output": {"beats": "bad"}})
        result = check_scene_output_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed

    def test_beat_missing_required(self):
        parsed = _make_parsed({"scene_output": {
            "beats": [{"speaker": "narrator"}],  # missing type and text
        }})
        result = check_scene_output_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed
        assert "type" in result.detail

    def test_beat_bad_visibility(self):
        parsed = _make_parsed({"scene_output": {
            "beats": [{"type": "narration", "text": "x", "visibility": "invalid"}],
        }})
        result = check_scene_output_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed
        assert "visibility" in result.detail

    def test_beat_list_field_not_list(self):
        parsed = _make_parsed({"scene_output": {
            "beats": [{"type": "narration", "text": "x", "actors": "guard"}],
        }})
        result = check_scene_output_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed
        assert "actors" in result.detail

    def test_no_beats_key(self):
        parsed = _make_parsed({"scene_output": {"other": "data"}})
        result = check_scene_output_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed  # legacy format OK


# ── dice_check validation ────────────────────────────────────────

class TestDiceCheckValid:
    def test_no_dice_check(self):
        parsed = _make_parsed({"narration": "hello"})
        result = check_dice_check_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_valid_dice_check(self):
        parsed = _make_parsed({"dice_check": {
            "attribute": "strength",
            "dc": 15,
            "context": "lift the boulder",
            "on_success": {"narration": "You heave it aside."},
            "on_failure": {"narration": "The boulder won't budge."},
        }})
        result = check_dice_check_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_missing_required(self):
        parsed = _make_parsed({"dice_check": {"attribute": "str"}})
        result = check_dice_check_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed
        assert "missing" in result.detail

    def test_dc_not_int(self):
        parsed = _make_parsed({"dice_check": {
            "attribute": "str", "dc": "hard",
            "context": "x",
            "on_success": {"narration": "ok"},
            "on_failure": {"narration": "fail"},
        }})
        result = check_dice_check_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed

    def test_on_success_missing_narration(self):
        parsed = _make_parsed({"dice_check": {
            "attribute": "str", "dc": 12,
            "context": "x",
            "on_success": {"state_update": {}},
            "on_failure": {"narration": "fail"},
        }})
        result = check_dice_check_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed
        assert "narration" in result.detail

    def test_not_dict(self):
        parsed = _make_parsed({"dice_check": "bad"})
        result = check_dice_check_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed


# ── puzzle_trigger validation ────────────────────────────────────

class TestPuzzleTriggerValid:
    def test_no_puzzle(self):
        parsed = _make_parsed({"narration": "hello"})
        result = check_puzzle_trigger_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_valid_puzzle(self):
        parsed = _make_parsed({"puzzle_trigger": {
            "puzzle_type": "riddle",
            "context": "The sphinx asks a riddle",
            "difficulty": "hard",
        }})
        result = check_puzzle_trigger_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_invalid_puzzle_type(self):
        parsed = _make_parsed({"puzzle_trigger": {
            "puzzle_type": "crossword",
            "context": "x",
        }})
        result = check_puzzle_trigger_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed
        assert "crossword" in result.detail

    def test_missing_required(self):
        parsed = _make_parsed({"puzzle_trigger": {"difficulty": "easy"}})
        result = check_puzzle_trigger_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed


# ── minigame_challenge validation ────────────────────────────────

class TestMinigameChallengeValid:
    def test_no_minigame(self):
        parsed = _make_parsed({"narration": "hello"})
        result = check_minigame_challenge_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_valid_minigame(self):
        parsed = _make_parsed({"minigame_challenge": {
            "game_type": "tic_tac_toe",
            "opponent_slug": "tavern-keeper",
            "stakes": "a free drink",
        }})
        result = check_minigame_challenge_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_invalid_game_type(self):
        parsed = _make_parsed({"minigame_challenge": {
            "game_type": "chess",
            "opponent_slug": "wizard",
        }})
        result = check_minigame_challenge_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed
        assert "chess" in result.detail

    def test_missing_required(self):
        parsed = _make_parsed({"minigame_challenge": {"stakes": "gold"}})
        result = check_minigame_challenge_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed


# ── game_time_advanced ───────────────────────────────────────────

class TestGameTimeAdvanced:
    def test_time_advanced(self):
        parsed = _make_parsed({
            "narration": "x",
            "state_update": {"game_time": {"day": 1, "hour": 10, "minute": 30}},
        })
        result = check_game_time_advanced(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_no_state_update(self):
        parsed = _make_parsed({"narration": "x"})
        result = check_game_time_advanced(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed

    def test_state_update_no_game_time(self):
        parsed = _make_parsed({
            "narration": "x",
            "state_update": {"mood": "dark"},
        })
        result = check_game_time_advanced(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed

    def test_tool_call_skipped(self):
        parsed = _make_parsed({"tool_call": "recent_turns"}, is_tool_call=True)
        result = check_game_time_advanced(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_game_time_not_dict(self):
        parsed = _make_parsed({
            "narration": "x",
            "state_update": {"game_time": "noon"},
        })
        result = check_game_time_advanced(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed


# ── state_no_character_removal ───────────────────────────────────

class TestStateNoCharacterRemoval:
    def test_no_state_update(self):
        parsed = _make_parsed({"narration": "x"})
        result = check_state_no_character_removal(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_clean_state_update(self):
        state = _make_state()
        state.characters["guard"] = {"name": "Guard"}
        parsed = _make_parsed({
            "state_update": {"mood": "tense"},
        })
        result = check_state_no_character_removal(parsed, _make_scenario(), TURN, state, P)
        assert result.passed

    def test_character_nulled_via_state(self):
        state = _make_state()
        state.characters["guard"] = {"name": "Guard"}
        parsed = _make_parsed({
            "state_update": {"guard": None},
        })
        result = check_state_no_character_removal(parsed, _make_scenario(), TURN, state, P)
        assert not result.passed
        assert "character_updates" in result.detail

    def test_non_character_null_ok(self):
        state = _make_state()
        state.characters["guard"] = {"name": "Guard"}
        parsed = _make_parsed({
            "state_update": {"old_event": None},
        })
        result = check_state_no_character_removal(parsed, _make_scenario(), TURN, state, P)
        assert result.passed


# ── inventory_changes_limit ──────────────────────────────────────

class TestInventoryChangesLimit:
    def test_no_player_update(self):
        parsed = _make_parsed({"narration": "x"})
        result = check_inventory_changes_limit(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_within_limit(self):
        parsed = _make_parsed({
            "player_state_update": {
                "inventory_add": ["sword", "shield"],
                "inventory_remove": ["stick"],
            },
        })
        result = check_inventory_changes_limit(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_exceeds_limit(self):
        parsed = _make_parsed({
            "player_state_update": {
                "inventory_add": [f"item-{i}" for i in range(8)],
                "inventory_remove": [f"old-{i}" for i in range(5)],
            },
        })
        result = check_inventory_changes_limit(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed
        assert "13" in result.detail

    def test_no_inventory_keys(self):
        parsed = _make_parsed({
            "player_state_update": {"hp": 50},
        })
        result = check_inventory_changes_limit(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed
