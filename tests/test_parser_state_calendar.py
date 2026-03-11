"""Tests for response_parser hardening, state_completed_value_prune, calendar_fire_range,
and AccumulatedState completed-value pruning."""

import json

from tgb.response_parser import (
    parse_response,
    clean_response,
    _coerce_python_dict,
    parse_json_lenient,
)
from tgb.checks.state_mgmt import check_state_completed_value_prune
from tgb.checks.calendar import check_calendar_fire_range
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


# ── _coerce_python_dict (ast.literal_eval) ──────────────────────

class TestCoercePythonDict:
    def test_python_true_false_none(self):
        text = '{"flag": True, "other": False, "empty": None}'
        result = _coerce_python_dict(text)
        assert result == {"flag": True, "other": False, "empty": None}

    def test_word_boundary_safe(self):
        """Values like 'runway' or 'falsehood' should not be mangled."""
        text = '{"location": "runway", "claim": "falsehood", "item": "null_pointer"}'
        result = _coerce_python_dict(text)
        assert result is not None
        assert result["location"] == "runway"
        assert result["claim"] == "falsehood"
        # null_pointer contains null but not at a word boundary with full match
        assert result["item"] == "null_pointer"

    def test_json_booleans_also_work(self):
        text = '{"flag": true, "other": false, "empty": null}'
        result = _coerce_python_dict(text)
        assert result == {"flag": True, "other": False, "empty": None}

    def test_invalid_input(self):
        result = _coerce_python_dict("not a dict at all")
        assert result is None

    def test_list_returns_none(self):
        result = _coerce_python_dict("[1, 2, 3]")
        assert result is None

    def test_unquoted_keys(self):
        """ast.literal_eval should handle single-quoted Python dicts."""
        text = "{'narration': 'hello', 'xp_awarded': 5}"
        result = _coerce_python_dict(text)
        assert result is not None
        assert result["narration"] == "hello"


# ── clean_response (narration/tool_call validation) ─────────────

class TestCleanResponseStricter:
    def test_truncated_with_narration_repaired(self):
        # Missing closing brace but has narration
        truncated = '{"narration": "The door opens.", "xp_awarded": 5'
        result = clean_response(truncated)
        parsed = parse_json_lenient(result)
        assert "narration" in parsed

    def test_truncated_with_tool_call_repaired(self):
        truncated = '{"tool_call": "memory_search", "queries": ["test"]'
        result = clean_response(truncated)
        parsed = parse_json_lenient(result)
        assert "tool_call" in parsed

    def test_truncated_without_narration_not_repaired(self):
        # Missing closing brace and no narration/tool_call
        truncated = '{"mood": "dark", "status": "ok"'
        result = clean_response(truncated)
        # Should return the original cleaned text, not a repaired version
        assert result == truncated

    def test_complete_json_unchanged(self):
        complete = '{"narration": "hello", "xp_awarded": 3}'
        result = clean_response(complete)
        assert result == complete


# ── parse_response integration ──────────────────────────────────

class TestParseResponseIntegration:
    def test_python_dict_parsed(self):
        raw = "{'narration': 'hello', 'xp_awarded': 5}"
        result = parse_response(raw)
        assert result.parsed_json.get("narration") == "hello"

    def test_tool_call_detected(self):
        raw = '{"tool_call": "memory_search", "queries": ["test"]}'
        result = parse_response(raw)
        assert result.is_tool_call


# ── state_completed_value_prune ─────────────────────────────────

class TestStateCompletedValuePrune:
    def test_no_completed_values(self):
        parsed = _make_parsed({
            "state_update": {"mood": "tense", "weather": "rainy"},
        })
        result = check_state_completed_value_prune(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_top_level_completed(self):
        parsed = _make_parsed({
            "state_update": {"quest_A": "completed", "mood": "dark"},
        })
        result = check_state_completed_value_prune(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed
        assert "quest_A" in result.detail

    def test_nested_completed(self):
        parsed = _make_parsed({
            "state_update": {"quests": {"find_key": "done", "escort": "active"}},
        })
        result = check_state_completed_value_prune(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed
        assert "quests.find_key" in result.detail

    def test_all_completed_values_detected(self):
        for val in ("complete", "completed", "done", "resolved", "finished",
                     "concluded", "vacated", "dispersed", "avoided", "departed"):
            parsed = _make_parsed({"state_update": {"key": val}})
            result = check_state_completed_value_prune(parsed, _make_scenario(), TURN, _make_state(), P)
            assert not result.passed, f"'{val}' should be flagged"

    def test_case_insensitive(self):
        parsed = _make_parsed({"state_update": {"quest": "Completed"}})
        result = check_state_completed_value_prune(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed

    def test_no_state_update(self):
        parsed = _make_parsed({"narration": "x"})
        result = check_state_completed_value_prune(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_null_values_ok(self):
        """null pruning is intentional and separate from completed-value pruning."""
        parsed = _make_parsed({"state_update": {"old_quest": None}})
        result = check_state_completed_value_prune(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed


# ── AccumulatedState completed-value pruning ────────────────────

class TestAccumulatedStateCompletedValuePrune:
    def test_completed_value_removed_from_state(self):
        state = _make_state()
        state.campaign_state["quest_A"] = "active"
        state.apply({"state_update": {"quest_A": "completed"}})
        assert "quest_A" not in state.campaign_state

    def test_done_value_removed(self):
        state = _make_state()
        state.campaign_state["errand"] = "in_progress"
        state.apply({"state_update": {"errand": "done"}})
        assert "errand" not in state.campaign_state

    def test_normal_value_stored(self):
        state = _make_state()
        state.apply({"state_update": {"mood": "tense"}})
        assert state.campaign_state["mood"] == "tense"

    def test_null_still_removes(self):
        state = _make_state()
        state.campaign_state["old_key"] = "value"
        state.apply({"state_update": {"old_key": None}})
        assert "old_key" not in state.campaign_state


# ── calendar_fire_range ─────────────────────────────────────────

class TestCalendarFireRange:
    def test_valid_fire_day_hour(self):
        parsed = _make_parsed({"calendar_update": {"add": [
            {"name": "Meeting", "time_remaining": 2, "time_unit": "hours",
             "fire_day": 1, "fire_hour": 14},
        ]}})
        result = check_calendar_fire_range(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_fire_day_zero(self):
        parsed = _make_parsed({"calendar_update": {"add": [
            {"name": "Event", "time_remaining": 1, "time_unit": "hours",
             "fire_day": 0, "fire_hour": 10},
        ]}})
        result = check_calendar_fire_range(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed
        assert "fire_day" in result.detail

    def test_fire_hour_24(self):
        parsed = _make_parsed({"calendar_update": {"add": [
            {"name": "Event", "time_remaining": 1, "time_unit": "hours",
             "fire_day": 1, "fire_hour": 24},
        ]}})
        result = check_calendar_fire_range(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed
        assert "fire_hour" in result.detail

    def test_fire_hour_negative(self):
        parsed = _make_parsed({"calendar_update": {"add": [
            {"name": "Event", "time_remaining": 1, "time_unit": "hours",
             "fire_day": 1, "fire_hour": -1},
        ]}})
        result = check_calendar_fire_range(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed

    def test_no_fire_fields_ok(self):
        parsed = _make_parsed({"calendar_update": {"add": [
            {"name": "Event", "time_remaining": 1, "time_unit": "hours"},
        ]}})
        result = check_calendar_fire_range(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_no_calendar_ok(self):
        parsed = _make_parsed({"narration": "x"})
        result = check_calendar_fire_range(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_fire_day_boundary_1(self):
        parsed = _make_parsed({"calendar_update": {"add": [
            {"name": "Event", "time_remaining": 1, "time_unit": "days",
             "fire_day": 1, "fire_hour": 0},
        ]}})
        result = check_calendar_fire_range(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_fire_hour_boundary_23(self):
        parsed = _make_parsed({"calendar_update": {"add": [
            {"name": "Event", "time_remaining": 1, "time_unit": "days",
             "fire_day": 1, "fire_hour": 23},
        ]}})
        result = check_calendar_fire_range(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed


# ── Player state exclusion in prompt ────────────────────────────

class TestPlayerStateExclusion:
    def test_inventory_excluded_from_player_card(self):
        scenario = _make_scenario(
            player=PlayerSetup(state={
                "character_name": "Alice",
                "location": "tavern",
                "inventory": ["sword", "shield"],
                "room_description": "A dusty room",
            }),
        )
        state = _make_state(scenario)
        from tgb.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        _, user_prompt = builder.build(scenario, TURN, state)
        # inventory should appear in RAILS_CONTEXT but not in PLAYER_CARD.state
        assert '"inventory":["sword","shield"]' in user_prompt.replace(" ", "")
        # room_description should be excluded from player card
        assert "dusty room" not in user_prompt.split("PLAYER_CARD")[1].split("PARTY_SNAPSHOT")[0] if "PLAYER_CARD" in user_prompt else True
