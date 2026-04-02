"""Tests for memory tool check functions."""

import pytest

from tgb.checks.memory import (
    check_memory_search_valid,
    check_memory_store_valid,
    check_memory_terms_valid,
    check_memory_turn_valid,
)
from tgb.checks.registry import CHECKS
from tgb.response_parser import ParsedResponse
from tgb.config import Scenario, CampaignSetup, PlayerSetup, TurnSpec
from tgb.prompt_builder import AccumulatedState


def _make_scenario(**kwargs) -> Scenario:
    return Scenario(
        name="test",
        description="test",
        tags=[],
        tier="basic",
        campaign=kwargs.get("campaign", CampaignSetup(name="test")),
        player=kwargs.get("player", PlayerSetup()),
        turns=[TurnSpec(action=kwargs.get("action", "test"))],
    )


def _make_state(scenario=None, **overrides) -> AccumulatedState:
    scenario = scenario or _make_scenario()
    state = AccumulatedState(scenario)
    for k, v in overrides.items():
        setattr(state, k, v)
    return state


def _make_parsed(json_data=None, raw="", is_tool_call=False) -> ParsedResponse:
    if json_data is None:
        json_data = {}
    if not raw:
        import json
        raw = json.dumps(json_data)
    return ParsedResponse(raw=raw, parsed_json=json_data, is_tool_call=is_tool_call)


class TestMemorySearchValid:
    def test_skip_non_memory_search(self):
        data = {"tool_call": "sms_read", "thread": "saul"}
        result = check_memory_search_valid(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed
        assert "skipped" in result.detail

    def test_valid_queries_list(self):
        data = {"tool_call": "memory_search", "queries": ["amber passage", "trade route"]}
        result = check_memory_search_valid(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_string_query_coerced(self):
        data = {"tool_call": "memory_search", "queries": "amber passage"}
        result = check_memory_search_valid(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_too_many_queries(self):
        data = {"tool_call": "memory_search", "queries": ["a", "b", "c", "d", "e"]}
        result = check_memory_search_valid(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "Too many queries" in result.detail

    def test_empty_queries_list(self):
        data = {"tool_call": "memory_search", "queries": []}
        result = check_memory_search_valid(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "empty" in result.detail

    def test_before_lines_out_of_range(self):
        data = {"tool_call": "memory_search", "queries": ["test"], "before_lines": 100}
        result = check_memory_search_valid(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "before_lines" in result.detail


class TestMemoryStoreValid:
    def test_skip_non_memory_store(self):
        data = {"tool_call": "memory_search", "queries": ["test"]}
        result = check_memory_store_valid(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed
        assert "skipped" in result.detail

    def test_valid_store(self):
        data = {
            "tool_call": "memory_store",
            "category": "research",
            "term": "amber-passage",
            "memory": "Found reference to Amber Passage in merchant diary.",
        }
        result = check_memory_store_valid(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_missing_category(self):
        data = {
            "tool_call": "memory_store",
            "memory": "Some memory text.",
        }
        result = check_memory_store_valid(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "category" in result.detail

    def test_missing_memory(self):
        data = {
            "tool_call": "memory_store",
            "category": "research",
        }
        result = check_memory_store_valid(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "memory" in result.detail

    def test_memory_too_long(self):
        data = {
            "tool_call": "memory_store",
            "category": "research",
            "memory": "x" * 1601,
        }
        result = check_memory_store_valid(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "1601" in result.detail


class TestMemoryTermsValid:
    def test_skip_non_memory_terms(self):
        data = {"tool_call": "memory_search", "queries": ["test"]}
        result = check_memory_terms_valid(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed
        assert "skipped" in result.detail

    def test_valid_with_wildcard(self):
        data = {"tool_call": "memory_terms", "wildcard": "*"}
        result = check_memory_terms_valid(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_wildcard_not_string(self):
        data = {"tool_call": "memory_terms", "wildcard": 123}
        result = check_memory_terms_valid(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "wildcard" in result.detail


class TestMemoryTurnValid:
    def test_skip_non_memory_turn(self):
        data = {"tool_call": "memory_search", "queries": ["test"]}
        result = check_memory_turn_valid(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed
        assert "skipped" in result.detail

    def test_valid_turn_id(self):
        data = {"tool_call": "memory_turn", "turn_id": 5}
        result = check_memory_turn_valid(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_missing_turn_id(self):
        data = {"tool_call": "memory_turn"}
        result = check_memory_turn_valid(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "turn_id" in result.detail


class TestRegistryMemory:
    def test_all_memory_checks_registered(self):
        for check_id in [
            "memory_search_valid",
            "memory_store_valid",
            "memory_terms_valid",
            "memory_turn_valid",
        ]:
            assert check_id in CHECKS, f"{check_id} not in registry"
