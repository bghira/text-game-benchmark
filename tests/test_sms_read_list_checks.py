"""Tests for SMS read/list check functions."""

import pytest

from tgb.checks.sms import (
    check_sms_read_valid,
    check_sms_list_valid,
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


class TestSmsReadValid:
    def test_skip_non_sms_read(self):
        data = {"tool_call": "sms_write", "thread": "saul", "from": "a", "to": "b", "message": "hi"}
        result = check_sms_read_valid(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed
        assert "skipped" in result.detail

    def test_valid_read(self):
        data = {"tool_call": "sms_read", "thread": "saul", "limit": 20}
        result = check_sms_read_valid(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_missing_thread(self):
        data = {"tool_call": "sms_read"}
        result = check_sms_read_valid(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "thread" in result.detail

    def test_limit_out_of_range(self):
        data = {"tool_call": "sms_read", "thread": "saul", "limit": 100}
        result = check_sms_read_valid(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "limit" in result.detail


class TestSmsListValid:
    def test_skip_non_sms_list(self):
        data = {"tool_call": "sms_read", "thread": "saul"}
        result = check_sms_list_valid(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed
        assert "skipped" in result.detail

    def test_valid_with_wildcard(self):
        data = {"tool_call": "sms_list", "wildcard": "*"}
        result = check_sms_list_valid(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed


class TestRegistrySmsReadList:
    def test_sms_read_registered(self):
        assert "sms_read_valid" in CHECKS

    def test_sms_list_registered(self):
        assert "sms_list_valid" in CHECKS
