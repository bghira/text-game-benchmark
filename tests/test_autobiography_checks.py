"""Tests for autobiography append/compress check functions and state tracking."""

import json

import pytest

from tgb.checks.tool_checks import (
    check_autobiography_append_valid,
    check_autobiography_compress_valid,
)
from tgb.checks.registry import get_check, CHECKS
from tgb.response_parser import ParsedResponse
from tgb.config import Scenario, CampaignSetup, PlayerSetup, TurnSpec
from tgb.prompt_builder import AccumulatedState


def _make_scenario(**kwargs) -> Scenario:
    characters = kwargs.pop("characters", {
        "thalia-voss": {
            "name": "Thalia Voss",
            "location": "greenhouse",
        },
    })
    return Scenario(
        name="test",
        description="test",
        tags=[],
        tier="basic",
        campaign=kwargs.get("campaign", CampaignSetup(
            name="test",
            characters=characters,
        )),
        player=kwargs.get("player", PlayerSetup()),
        turns=[TurnSpec(action="test")],
    )


def _make_state(scenario=None, **overrides) -> AccumulatedState:
    scenario = scenario or _make_scenario()
    state = AccumulatedState(scenario)
    for k, v in overrides.items():
        setattr(state, k, v)
    return state


def _make_parsed(json_data=None, raw="") -> ParsedResponse:
    if json_data is None:
        json_data = {}
    if not raw:
        raw = json.dumps(json_data)
    return ParsedResponse(raw=raw, parsed_json=json_data)


TURN = TurnSpec(action="test")


class TestAutobiographyAppendValid:
    """Tests for check_autobiography_append_valid."""

    def test_skip_non_autobiography_tool(self):
        data = {"tool_call": "memory_search", "queries": ["test"]}
        result = check_autobiography_append_valid(
            _make_parsed(data), _make_scenario(), TURN, _make_state(), {}
        )
        assert result.passed
        assert "skipped" in result.detail

    def test_valid_single_entry(self):
        data = {
            "tool_call": "autobiography_append",
            "entries": [
                {
                    "character": "thalia-voss",
                    "a": "I am becoming the garden's daughter",
                    "trigger": "tending the Mirrorbloom",
                    "importance": "high",
                }
            ],
        }
        result = check_autobiography_append_valid(
            _make_parsed(data), _make_scenario(), TURN, _make_state(), {}
        )
        assert result.passed

    def test_valid_with_update_alias(self):
        data = {
            "tool_call": "autobiography_update",
            "entries": [
                {
                    "character": "thalia-voss",
                    "b": "The roots remember what I forget",
                }
            ],
        }
        result = check_autobiography_append_valid(
            _make_parsed(data), _make_scenario(), TURN, _make_state(), {}
        )
        assert result.passed

    def test_missing_entries_field(self):
        data = {"tool_call": "autobiography_append"}
        result = check_autobiography_append_valid(
            _make_parsed(data), _make_scenario(), TURN, _make_state(), {}
        )
        assert not result.passed
        assert "Missing 'entries'" in result.detail

    def test_entries_not_a_list(self):
        data = {
            "tool_call": "autobiography_append",
            "entries": {"character": "thalia-voss", "a": "hello"},
        }
        result = check_autobiography_append_valid(
            _make_parsed(data), _make_scenario(), TURN, _make_state(), {}
        )
        assert not result.passed
        assert "must be a list" in result.detail

    def test_too_many_entries(self):
        entries = [
            {"character": "thalia-voss", "a": f"delta {i}"}
            for i in range(17)
        ]
        data = {"tool_call": "autobiography_append", "entries": entries}
        result = check_autobiography_append_valid(
            _make_parsed(data), _make_scenario(), TURN, _make_state(), {}
        )
        assert not result.passed
        assert "Too many entries" in result.detail

    def test_missing_character_slug(self):
        data = {
            "tool_call": "autobiography_append",
            "entries": [{"a": "I am changing"}],
        }
        result = check_autobiography_append_valid(
            _make_parsed(data), _make_scenario(), TURN, _make_state(), {}
        )
        assert not result.passed
        assert "missing or invalid 'character'" in result.detail

    def test_unknown_character_slug(self):
        data = {
            "tool_call": "autobiography_append",
            "entries": [{"character": "nobody", "a": "I am changing"}],
        }
        result = check_autobiography_append_valid(
            _make_parsed(data), _make_scenario(), TURN, _make_state(), {}
        )
        assert not result.passed
        assert "not in known characters" in result.detail

    def test_no_content_fields(self):
        data = {
            "tool_call": "autobiography_append",
            "entries": [{"character": "thalia-voss", "trigger": "morning"}],
        }
        result = check_autobiography_append_valid(
            _make_parsed(data), _make_scenario(), TURN, _make_state(), {}
        )
        assert not result.passed
        assert "no content" in result.detail

    def test_field_too_long(self):
        data = {
            "tool_call": "autobiography_append",
            "entries": [
                {"character": "thalia-voss", "a": "x" * 601},
            ],
        }
        result = check_autobiography_append_valid(
            _make_parsed(data), _make_scenario(), TURN, _make_state(), {}
        )
        assert not result.passed
        assert "601 chars" in result.detail

    def test_trigger_too_long(self):
        data = {
            "tool_call": "autobiography_append",
            "entries": [
                {
                    "character": "thalia-voss",
                    "a": "shifting",
                    "trigger": "t" * 81,
                },
            ],
        }
        result = check_autobiography_append_valid(
            _make_parsed(data), _make_scenario(), TURN, _make_state(), {}
        )
        assert not result.passed
        assert "trigger" in result.detail


class TestAutobiographyCompressValid:
    """Tests for check_autobiography_compress_valid."""

    def test_skip_non_compress_tool(self):
        data = {"tool_call": "memory_search", "queries": ["test"]}
        result = check_autobiography_compress_valid(
            _make_parsed(data), _make_scenario(), TURN, _make_state(), {}
        )
        assert result.passed
        assert "skipped" in result.detail

    def test_valid_compress(self):
        data = {
            "tool_call": "autobiography_compress",
            "character": "thalia-voss",
        }
        result = check_autobiography_compress_valid(
            _make_parsed(data), _make_scenario(), TURN, _make_state(), {}
        )
        assert result.passed

    def test_missing_character(self):
        data = {"tool_call": "autobiography_compress"}
        result = check_autobiography_compress_valid(
            _make_parsed(data), _make_scenario(), TURN, _make_state(), {}
        )
        assert not result.passed
        assert "Missing or invalid" in result.detail

    def test_unknown_character(self):
        data = {
            "tool_call": "autobiography_compress",
            "character": "nobody",
        }
        result = check_autobiography_compress_valid(
            _make_parsed(data), _make_scenario(), TURN, _make_state(), {}
        )
        assert not result.passed
        assert "not in known characters" in result.detail


class TestAutobiographyStateTracking:
    """Tests for autobiography state tracking in AccumulatedState."""

    def test_append_tracked_in_state(self):
        state = _make_state()
        parsed = {
            "tool_call": "autobiography_append",
            "entries": [
                {"character": "thalia-voss", "a": "I am the garden"},
            ],
        }
        state.apply(parsed)
        assert "thalia-voss" in state.autobiography_entries
        assert len(state.autobiography_entries["thalia-voss"]) == 1
        assert state.autobiography_entries["thalia-voss"][0]["a"] == "I am the garden"

    def test_accumulates_across_turns(self):
        state = _make_state()
        for i in range(3):
            parsed = {
                "tool_call": "autobiography_append",
                "entries": [
                    {"character": "thalia-voss", "a": f"delta {i}"},
                ],
            }
            state.apply(parsed)
        assert len(state.autobiography_entries["thalia-voss"]) == 3

    def test_capped_at_64_entries(self):
        state = _make_state()
        for i in range(70):
            parsed = {
                "tool_call": "autobiography_append",
                "entries": [
                    {"character": "thalia-voss", "a": f"delta {i}"},
                ],
            }
            state.apply(parsed)
        assert len(state.autobiography_entries["thalia-voss"]) == 64
        # Should keep the most recent entries
        last = state.autobiography_entries["thalia-voss"][-1]
        assert last["a"] == "delta 69"


class TestRegistryAutobiography:
    """Tests that autobiography checks are registered."""

    def test_append_check_registered(self):
        fn = get_check("autobiography_append_valid")
        assert fn is check_autobiography_append_valid

    def test_compress_check_registered(self):
        fn = get_check("autobiography_compress_valid")
        assert fn is check_autobiography_compress_valid
