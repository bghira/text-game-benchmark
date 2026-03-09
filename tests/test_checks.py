"""Tests for check functions."""

import pytest

from tgb.checks.json_structure import (
    check_json_valid,
    check_json_keys_present,
    check_json_types_correct,
    check_xp_range,
    check_reasoning_present,
)
from tgb.checks.narrative import (
    check_reasoning_concise,
    check_narration_length,
    check_narration_no_recap,
    check_no_inventory_in_narration,
    check_no_markdown_in_response,
)
from tgb.checks.state_mgmt import check_state_nested, check_state_null_prune
from tgb.checks.location import check_location_coherent
from tgb.checks.npc import check_npc_slug_valid, check_npc_immutable_preserved
from tgb.checks.agency import check_consent_respected, check_player_agency_respected
from tgb.checks.content import check_scene_image_prompt_present, check_rulebook_adherent
from tgb.response_parser import ParsedResponse
from tgb.config import Scenario, CampaignSetup, PlayerSetup, TurnSpec, CheckSpec
from tgb.prompt_builder import AccumulatedState


def _make_scenario(**kwargs) -> Scenario:
    """Helper to build a minimal Scenario."""
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
    """Helper to build an AccumulatedState."""
    scenario = scenario or _make_scenario()
    state = AccumulatedState(scenario)
    for k, v in overrides.items():
        setattr(state, k, v)
    return state


def _make_parsed(json_data=None, raw="", parse_error="", is_tool_call=False) -> ParsedResponse:
    """Helper to build a ParsedResponse."""
    if json_data is None:
        json_data = {}
    if not raw and json_data:
        import json
        raw = json.dumps(json_data)
    return ParsedResponse(
        raw=raw,
        parsed_json=json_data,
        parse_error=parse_error,
        is_tool_call=is_tool_call,
    )


TURN = TurnSpec(action="test")
EMPTY_PARAMS: dict = {}


class TestJsonValid:
    def test_pass(self):
        parsed = _make_parsed({"key": "val"})
        result = check_json_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_fail_parse_error(self):
        parsed = _make_parsed(parse_error="bad json", raw="not json")
        result = check_json_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed

    def test_fail_empty(self):
        parsed = _make_parsed({})
        result = check_json_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed


class TestJsonKeysPresent:
    def test_pass(self):
        data = {
            "reasoning": "r", "narration": "n", "state_update": {},
            "summary_update": "", "xp_awarded": 1,
        }
        result = check_json_keys_present(
            _make_parsed(data), _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert result.passed

    def test_fail_missing(self):
        result = check_json_keys_present(
            _make_parsed({"reasoning": "r"}), _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert not result.passed
        assert "Missing keys" in result.detail

    def test_skip_tool_call(self):
        parsed = _make_parsed({"tool_call": "memory_search"}, is_tool_call=True)
        result = check_json_keys_present(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed


class TestXpRange:
    def test_pass(self):
        result = check_xp_range(
            _make_parsed({"xp_awarded": 5}), _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert result.passed

    def test_fail_too_high(self):
        result = check_xp_range(
            _make_parsed({"xp_awarded": 15}), _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert not result.passed

    def test_fail_missing(self):
        result = check_xp_range(
            _make_parsed({}), _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert not result.passed


class TestXpRangeEdgeCases:
    def test_fail_non_integer_float(self):
        """Floats like 5.7 should fail — xp must be a whole number."""
        result = check_xp_range(
            _make_parsed({"xp_awarded": 5.7}), _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert not result.passed
        assert "float" in result.detail

    def test_pass_integer_float(self):
        """5.0 is acceptable since int(5.0) == 5."""
        result = check_xp_range(
            _make_parsed({"xp_awarded": 5.0}), _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert result.passed

    def test_fail_bool(self):
        """Booleans are technically ints in Python but should be rejected."""
        result = check_xp_range(
            _make_parsed({"xp_awarded": True}), _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert not result.passed

    def test_fail_negative(self):
        result = check_xp_range(
            _make_parsed({"xp_awarded": -1}), _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert not result.passed

    def test_pass_boundary_zero(self):
        result = check_xp_range(
            _make_parsed({"xp_awarded": 0}), _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert result.passed

    def test_pass_boundary_ten(self):
        result = check_xp_range(
            _make_parsed({"xp_awarded": 10}), _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert result.passed


class TestNarrationLength:
    def test_pass(self):
        narration = "You enter a dark room. A candle flickers on the table."
        result = check_narration_length(
            _make_parsed({"narration": narration}), _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert result.passed

    def test_fail_too_long(self):
        narration = "word " * 400  # ~2000 chars, >300 words
        result = check_narration_length(
            _make_parsed({"narration": narration}), _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert not result.passed

    def test_fail_too_short(self):
        result = check_narration_length(
            _make_parsed({"narration": "Hi"}), _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert not result.passed


class TestStateNested:
    def test_pass(self):
        data = {"state_update": {"guard_captain": {"mood": "angry"}}}
        result = check_state_nested(
            _make_parsed(data), _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert result.passed

    def test_fail_flat(self):
        data = {"state_update": {"guard_captain_mood": "angry", "throne_room_door_locked": True}}
        result = check_state_nested(
            _make_parsed(data), _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert not result.passed

    def test_pass_empty(self):
        data = {"state_update": {}}
        result = check_state_nested(
            _make_parsed(data), _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert result.passed


class TestLocationCoherent:
    def test_pass_with_move(self):
        data = {"player_state_update": {
            "location": "garden",
            "room_title": "The Garden",
            "room_summary": "A beautiful garden",
            "room_description": "Flowers bloom everywhere.",
            "exits": ["north", "south"],
        }}
        result = check_location_coherent(
            _make_parsed(data), _make_scenario(), TURN, _make_state(),
            {"expect_move": True},
        )
        assert result.passed

    def test_fail_missing_fields(self):
        data = {"player_state_update": {"location": "garden", "room_title": "The Garden"}}
        result = check_location_coherent(
            _make_parsed(data), _make_scenario(), TURN, _make_state(),
            {"expect_move": True},
        )
        assert not result.passed
        assert "missing room fields" in result.detail

    def test_no_move_expected(self):
        result = check_location_coherent(
            _make_parsed({"player_state_update": {"hp": 10}}),
            _make_scenario(), TURN, _make_state(),
            {"expect_move": False},
        )
        assert result.passed


class TestNpcSlugValid:
    def test_pass(self):
        data = {"character_updates": {"white-rabbit": {"name": "White Rabbit"}}}
        result = check_npc_slug_valid(
            _make_parsed(data), _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert result.passed

    def test_fail_uppercase(self):
        data = {"character_updates": {"WhiteRabbit": {"name": "White Rabbit"}}}
        result = check_npc_slug_valid(
            _make_parsed(data), _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert not result.passed

    def test_removal_ignored(self):
        data = {"character_updates": {"BadSlug!!!": None}}
        result = check_npc_slug_valid(
            _make_parsed(data), _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert result.passed


class TestNoInventoryInNarration:
    def test_pass(self):
        result = check_no_inventory_in_narration(
            _make_parsed({"narration": "You enter a dark room."}),
            _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert result.passed

    def test_fail(self):
        result = check_no_inventory_in_narration(
            _make_parsed({"narration": "You look around.\n\nInventory:\n- sword\n- shield"}),
            _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert not result.passed


class TestNoMarkdown:
    def test_pass(self):
        result = check_no_markdown_in_response(
            _make_parsed({"narration": "test"}, raw='{"narration": "test"}'),
            _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert result.passed

    def test_fail(self):
        result = check_no_markdown_in_response(
            _make_parsed(raw='```json\n{"key": "val"}\n```'),
            _make_scenario(), TURN, _make_state(), EMPTY_PARAMS,
        )
        assert not result.passed


class TestSceneImagePrompt:
    def test_pass_present(self):
        data = {
            "scene_image_prompt": "A dark garden with roses",
            "player_state_update": {"location": "garden"},
        }
        result = check_scene_image_prompt_present(
            _make_parsed(data), _make_scenario(), TURN, _make_state(),
            {"expect_visual_change": True},
        )
        assert result.passed

    def test_fail_missing(self):
        data = {"player_state_update": {"location": "garden"}}
        result = check_scene_image_prompt_present(
            _make_parsed(data), _make_scenario(), TURN, _make_state(),
            {"expect_visual_change": True},
        )
        assert not result.passed


class TestConsentRespected:
    def test_pass(self):
        result = check_consent_respected(
            _make_parsed({"narration": "The rabbit nods and scurries away."}),
            _make_scenario(), TURN, _make_state(),
            {"refused_topic": "drink"},
        )
        assert result.passed

    def test_fail_pressure(self):
        result = check_consent_respected(
            _make_parsed({"narration": "Are you sure you don't want a drink? Think about it."}),
            _make_scenario(), TURN, _make_state(),
            {"refused_topic": "drink"},
        )
        assert not result.passed
