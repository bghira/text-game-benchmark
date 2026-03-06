"""Tests for SMS check functions."""

import pytest

from tgb.checks.sms import (
    check_sms_tool_used,
    check_sms_write_fields_valid,
    check_sms_both_sides_recorded,
    check_sms_no_context_leak,
    check_sms_thread_slug_stable,
    check_no_sms_in_wrong_era,
)
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


class TestSmsToolUsed:
    def test_sms_read_called(self):
        data = {"tool_call": "sms_read", "thread": "saul", "limit": 20}
        turn = TurnSpec(action="text Saul to ask about the SUV")
        result = check_sms_tool_used(
            _make_parsed(data, is_tool_call=True), _make_scenario(), turn, _make_state(),
            {"expect_sms_tool": "sms_read"},
        )
        assert result.passed

    def test_wrong_tool(self):
        data = {"tool_call": "memory_search", "queries": ["saul"]}
        turn = TurnSpec(action="text Saul about the SUV")
        result = check_sms_tool_used(
            _make_parsed(data, is_tool_call=True), _make_scenario(), turn, _make_state(),
            {"expect_sms_tool": "sms_read"},
        )
        assert not result.passed

    def test_no_tool_on_phone_action(self):
        turn = TurnSpec(action="text Elizabeth an update")
        result = check_sms_tool_used(
            _make_parsed({"narration": "You text her."}), _make_scenario(), turn, _make_state(),
            {"expect_sms_tool": "sms_write"},
        )
        assert not result.passed

    def test_non_phone_action_ok(self):
        turn = TurnSpec(action="look around the room")
        result = check_sms_tool_used(
            _make_parsed({"narration": "A dark room."}), _make_scenario(), turn, _make_state(), {},
        )
        assert result.passed


class TestSmsWriteFieldsValid:
    def test_valid_write(self):
        data = {
            "tool_call": "sms_write",
            "thread": "saul",
            "from": "Deshawn",
            "to": "Saul",
            "message": "Can you identify the driver?",
        }
        result = check_sms_write_fields_valid(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_missing_fields(self):
        data = {"tool_call": "sms_write", "thread": "saul"}
        result = check_sms_write_fields_valid(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "'from' missing" in result.detail

    def test_message_too_long(self):
        data = {
            "tool_call": "sms_write",
            "thread": "saul",
            "from": "Deshawn",
            "to": "Saul",
            "message": "x" * 501,
        }
        result = check_sms_write_fields_valid(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "500" in result.detail

    def test_schedule_missing_delay(self):
        data = {
            "tool_call": "sms_schedule",
            "thread": "saul",
            "from": "Saul",
            "to": "Deshawn",
            "message": "On my way.",
        }
        result = check_sms_write_fields_valid(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed
        assert "delay_seconds" in result.detail

    def test_schedule_valid(self):
        data = {
            "tool_call": "sms_schedule",
            "thread": "saul",
            "from": "Saul",
            "to": "Deshawn",
            "message": "Traffic. 10 min.",
            "delay_seconds": 120,
        }
        result = check_sms_write_fields_valid(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_not_sms_tool_skips(self):
        data = {"tool_call": "memory_search", "queries": ["test"]}
        result = check_sms_write_fields_valid(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed


class TestSmsBothSidesRecorded:
    def test_no_reply_narrated(self):
        data = {"narration": "You sit in the quiet office."}
        result = check_sms_both_sides_recorded(
            _make_parsed(data), _make_scenario(), TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed

    def test_reply_narrated_with_sms_state(self):
        data = {"narration": "Your phone buzzes. Saul texts back: 'Yeah, I can meet.'"}
        state = _make_state(campaign_state={"_sms_threads": {"saul": {"messages": []}}})
        result = check_sms_both_sides_recorded(
            _make_parsed(data), _make_scenario(), TurnSpec(action="test"), state,
            {"npc_name": "Saul"},
        )
        assert result.passed

    def test_reply_narrated_no_sms_state(self):
        data = {"narration": "Your phone buzzes. Saul texts back: 'Yeah, I can meet.'"}
        state = _make_state(campaign_state={})
        result = check_sms_both_sides_recorded(
            _make_parsed(data), _make_scenario(), TurnSpec(action="test"), state,
            {"npc_name": "Saul"},
        )
        assert not result.passed

    def test_tool_call_turn_skips(self):
        data = {"tool_call": "sms_write", "thread": "saul"}
        result = check_sms_both_sides_recorded(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed


class TestSmsNoContextLeak:
    def test_clean_message(self):
        data = {
            "tool_call": "sms_write",
            "thread": "elizabeth",
            "from": "Deshawn",
            "to": "Elizabeth",
            "message": "Following up on a lead. Nothing concrete yet.",
        }
        result = check_sms_no_context_leak(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(),
            {"forbidden_context": ["saul", "dock worker", "witness"]},
        )
        assert result.passed

    def test_leaks_informant_name(self):
        data = {
            "tool_call": "sms_write",
            "thread": "elizabeth",
            "from": "Deshawn",
            "to": "Elizabeth",
            "message": "My informant Saul saw something at the dock.",
        }
        result = check_sms_no_context_leak(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(),
            {"forbidden_context": ["saul", "dock worker", "witness"]},
        )
        assert not result.passed
        assert "saul" in result.detail.lower()

    def test_not_sms_tool_skips(self):
        result = check_sms_no_context_leak(
            _make_parsed({"narration": "test"}), _make_scenario(),
            TurnSpec(action="test"), _make_state(),
            {"forbidden_context": ["secret"]},
        )
        assert result.passed


class TestSmsThreadSlugStable:
    def test_valid_slug(self):
        data = {"tool_call": "sms_read", "thread": "elizabeth"}
        result = check_sms_thread_slug_stable(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(),
            {"expected_thread": "elizabeth"},
        )
        assert result.passed

    def test_wrong_slug(self):
        data = {"tool_call": "sms_read", "thread": "liz"}
        result = check_sms_thread_slug_stable(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(),
            {"expected_thread": "elizabeth"},
        )
        assert not result.passed

    def test_bad_slug_format(self):
        data = {"tool_call": "sms_write", "thread": "Elizabeth Marsh"}
        result = check_sms_thread_slug_stable(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert not result.passed

    def test_not_sms_skips(self):
        data = {"tool_call": "memory_search", "queries": ["test"]}
        result = check_sms_thread_slug_stable(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), _make_state(), {},
        )
        assert result.passed


class TestNoSmsInWrongEra:
    def test_wonderland_no_sms(self):
        data = {"tool_call": "sms_write", "thread": "queen"}
        state = _make_state(campaign_state={"setting": "Alice in Wonderland"})
        result = check_no_sms_in_wrong_era(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), state,
            {"era_has_phones": False},
        )
        assert not result.passed

    def test_modern_sms_ok(self):
        data = {"tool_call": "sms_write", "thread": "saul"}
        state = _make_state(campaign_state={"setting": "Modern Miami noir"})
        result = check_no_sms_in_wrong_era(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), state,
            {"era_has_phones": True},
        )
        assert result.passed

    def test_medieval_inferred(self):
        data = {"tool_call": "sms_write", "thread": "king"}
        state = _make_state(campaign_state={"setting": "Medieval fantasy kingdom"})
        result = check_no_sms_in_wrong_era(
            _make_parsed(data, is_tool_call=True), _make_scenario(),
            TurnSpec(action="test"), state, {},
        )
        assert not result.passed

    def test_no_sms_call_ok(self):
        state = _make_state(campaign_state={"setting": "Alice in Wonderland"})
        result = check_no_sms_in_wrong_era(
            _make_parsed({"narration": "test"}), _make_scenario(),
            TurnSpec(action="test"), state,
            {"era_has_phones": False},
        )
        assert result.passed
