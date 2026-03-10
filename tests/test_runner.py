"""Tests for runner.py — tool-call loop and synthetic results."""

import json

from tgb.config import Scenario, CampaignSetup, PlayerSetup, TurnSpec
from tgb.prompt_builder import AccumulatedState
from tgb.runner import _synthetic_tool_result, _merge_timing, MAX_TOOL_ROUNDS
from tgb.clients.ollama_client import TimingData


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


def _make_state(scenario=None) -> AccumulatedState:
    scenario = scenario or _make_scenario()
    return AccumulatedState(scenario)


class TestSyntheticToolResult:
    def test_recent_turns(self):
        state = _make_state()
        state.recent_turns = [
            {"tag": "[TURN #1]", "action": "look", "narration": "You see a room."},
        ]
        result = json.loads(_synthetic_tool_result({"tool_call": "recent_turns"}, state))
        assert result["tool_result"] == "recent_turns"
        assert len(result["turns"]) == 1
        assert result["turns"][0]["action"] == "look"

    def test_ready_to_write(self):
        state = _make_state()
        result = json.loads(_synthetic_tool_result({"tool_call": "ready_to_write"}, state))
        assert result["tool_result"] == "ready_to_write"
        assert result["status"] == "ok"

    def test_memory_search(self):
        state = _make_state()
        result = json.loads(_synthetic_tool_result({"tool_call": "memory_search"}, state))
        assert result["tool_result"] == "memory_search"
        assert result["results"] == []

    def test_sms_write(self):
        state = _make_state()
        result = json.loads(_synthetic_tool_result({"tool_call": "sms_write"}, state))
        assert result["tool_result"] == "sms_write"
        assert result["status"] == "sent"

    def test_sms_schedule(self):
        state = _make_state()
        result = json.loads(_synthetic_tool_result({"tool_call": "sms_schedule"}, state))
        assert result["status"] == "scheduled"

    def test_sms_list(self):
        state = _make_state()
        result = json.loads(_synthetic_tool_result({"tool_call": "sms_list"}, state))
        assert result["tool_result"] == "sms_list"

    def test_source_browse(self):
        state = _make_state()
        result = json.loads(_synthetic_tool_result({"tool_call": "source_browse"}, state))
        assert result["tool_result"] == "source_browse"

    def test_name_generate(self):
        state = _make_state()
        result = json.loads(_synthetic_tool_result({"tool_call": "name_generate"}, state))
        assert result["tool_result"] == "name_generate"
        assert len(result["names"]) == 1

    def test_plot_plan(self):
        state = _make_state()
        result = json.loads(_synthetic_tool_result({"tool_call": "plot_plan"}, state))
        assert result["status"] == "recorded"

    def test_unknown_tool(self):
        state = _make_state()
        result = json.loads(_synthetic_tool_result({"tool_call": "unknown_widget"}, state))
        assert result["tool_result"] == "unknown_widget"
        assert result["status"] == "ok"

    def test_recent_turns_limits_to_8(self):
        state = _make_state()
        state.recent_turns = [
            {"tag": f"[TURN #{i}]", "action": f"act-{i}", "narration": f"nar-{i}"}
            for i in range(20)
        ]
        result = json.loads(_synthetic_tool_result({"tool_call": "recent_turns"}, state))
        assert len(result["turns"]) == 8


class TestMergeTiming:
    def test_accumulates(self):
        t1 = TimingData(prompt_tokens=100, completion_tokens=50, wall_clock_seconds=1.0)
        t2 = TimingData(prompt_tokens=200, completion_tokens=80, wall_clock_seconds=2.0,
                        eval_tokens_per_sec=40.0)
        merged = _merge_timing(t1, t2)
        assert merged.prompt_tokens == 300
        assert merged.completion_tokens == 130
        assert merged.wall_clock_seconds == 3.0
        assert merged.eval_tokens_per_sec == 40.0

    def test_none_fields(self):
        t1 = TimingData()
        t2 = TimingData(prompt_tokens=100, wall_clock_seconds=1.0)
        merged = _merge_timing(t1, t2)
        assert merged.prompt_tokens == 100
        assert merged.wall_clock_seconds == 1.0


class TestMaxToolRounds:
    def test_constant_value(self):
        assert MAX_TOOL_ROUNDS == 4
