"""Tests for SMS state tracking, CURRENTLY_ATTENTIVE_PLAYERS, name_generate,
communication_rules, and chapter close action."""

import json

from tgb.checks.tool_checks import (
    check_tool_format_valid,
    check_communication_rules_valid,
    check_name_generate_valid,
)
from tgb.checks.subplot import chapter_fields_valid
from tgb.response_parser import ParsedResponse
from tgb.config import Scenario, CampaignSetup, PlayerSetup, TurnSpec
from tgb.prompt_builder import AccumulatedState, PromptBuilder


def _make_scenario(**kwargs) -> Scenario:
    return Scenario(
        name=kwargs.get("name", "test"),
        description="test",
        tags=[],
        tier="basic",
        campaign=kwargs.get("campaign", CampaignSetup(name="test")),
        player=kwargs.get("player", PlayerSetup()),
        turns=[TurnSpec(action="test")],
        attentive_players=kwargs.get("attentive_players", []),
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


# ── SMS state tracking in AccumulatedState ─────────────────────


class TestSmsStateTracking:
    def test_sms_write_creates_thread(self):
        state = _make_state()
        state.apply({
            "tool_call": "sms_write",
            "thread": "elizabeth",
            "from": "Deshawn",
            "to": "Elizabeth",
            "message": "Hey, are you there?",
        })
        threads = state.campaign_state.get("_sms_threads", {})
        assert "elizabeth" in threads
        assert len(threads["elizabeth"]["messages"]) == 1
        msg = threads["elizabeth"]["messages"][0]
        assert msg["from"] == "Deshawn"
        assert msg["to"] == "Elizabeth"
        assert msg["message"] == "Hey, are you there?"

    def test_sms_write_appends_to_thread(self):
        state = _make_state()
        state.apply({
            "tool_call": "sms_write",
            "thread": "elizabeth",
            "from": "Deshawn",
            "to": "Elizabeth",
            "message": "First message",
        })
        state.apply({
            "tool_call": "sms_write",
            "thread": "elizabeth",
            "from": "Elizabeth",
            "to": "Deshawn",
            "message": "Reply message",
        })
        threads = state.campaign_state["_sms_threads"]
        assert len(threads["elizabeth"]["messages"]) == 2
        assert threads["elizabeth"]["messages"][1]["from"] == "Elizabeth"

    def test_sms_write_tracks_sequence(self):
        state = _make_state()
        state.apply({
            "tool_call": "sms_write",
            "thread": "alice",
            "from": "Bob",
            "to": "Alice",
            "message": "Hi",
        })
        state.apply({
            "tool_call": "sms_write",
            "thread": "charlie",
            "from": "Bob",
            "to": "Charlie",
            "message": "Hey",
        })
        seq1 = state.campaign_state["_sms_threads"]["alice"]["messages"][0]["seq"]
        seq2 = state.campaign_state["_sms_threads"]["charlie"]["messages"][0]["seq"]
        assert seq2 > seq1

    def test_sms_schedule_also_tracked(self):
        state = _make_state()
        state.apply({
            "tool_call": "sms_schedule",
            "thread": "mom",
            "from": "NPC",
            "to": "Player",
            "message": "Don't forget dinner!",
            "delay_seconds": 300,
        })
        threads = state.campaign_state.get("_sms_threads", {})
        assert "mom" in threads
        assert len(threads["mom"]["messages"]) == 1

    def test_sms_thread_key_normalized(self):
        """Thread keys are lowercased."""
        state = _make_state()
        state.apply({
            "tool_call": "sms_write",
            "thread": "Elizabeth",
            "from": "Deshawn",
            "to": "Elizabeth",
            "message": "Test",
        })
        threads = state.campaign_state["_sms_threads"]
        assert "elizabeth" in threads
        assert "Elizabeth" not in threads

    def test_sms_max_messages_enforced(self):
        state = _make_state()
        # Send 45 messages (limit is 40)
        for i in range(45):
            state.apply({
                "tool_call": "sms_write",
                "thread": "spam",
                "from": "A",
                "to": "B",
                "message": f"Message {i}",
            })
        messages = state.campaign_state["_sms_threads"]["spam"]["messages"]
        assert len(messages) == AccumulatedState.SMS_MAX_MESSAGES_PER_THREAD
        # Oldest messages should be truncated
        assert "Message 44" in messages[-1]["message"]

    def test_sms_missing_fields_ignored(self):
        """SMS write with missing required fields is silently ignored."""
        state = _make_state()
        state.apply({
            "tool_call": "sms_write",
            "thread": "test",
            "from": "",
            "to": "Bob",
            "message": "Hi",
        })
        # No thread created because sender is empty
        threads = state.campaign_state.get("_sms_threads", {})
        assert "test" not in threads

    def test_sms_field_truncation(self):
        """Fields are truncated to engine limits."""
        state = _make_state()
        long_name = "A" * 200
        state.apply({
            "tool_call": "sms_write",
            "thread": "test",
            "from": long_name,
            "to": "Bob",
            "message": "Hi",
        })
        threads = state.campaign_state["_sms_threads"]
        msg = threads["test"]["messages"][0]
        assert len(msg["from"]) == 80


# ── CURRENTLY_ATTENTIVE_PLAYERS ─────────────────────────────────


class TestAttentivePlayers:
    def test_empty_by_default(self):
        scenario = _make_scenario()
        state = _make_state(scenario)
        builder = PromptBuilder()
        _, user_prompt = builder.build(scenario, TURN, state)
        assert "CURRENTLY_ATTENTIVE_PLAYERS:[]" in user_prompt.replace(" ", "")

    def test_populated_from_scenario(self):
        attentive = [
            {"actor_id": 1, "name": "Alice", "player_slug": "alice", "seconds_since_last_message": 30},
        ]
        scenario = _make_scenario(attentive_players=attentive)
        state = _make_state(scenario)
        builder = PromptBuilder()
        _, user_prompt = builder.build(scenario, TURN, state)
        assert '"actor_id":1' in user_prompt.replace(" ", "")
        assert '"player_slug":"alice"' in user_prompt.replace(" ", "")

    def test_attention_window_present(self):
        scenario = _make_scenario()
        state = _make_state(scenario)
        builder = PromptBuilder()
        _, user_prompt = builder.build(scenario, TURN, state)
        assert "ATTENTION_WINDOW_SECONDS: 600" in user_prompt


# ── communication_rules_valid ───────────────────────────────────


class TestCommunicationRulesValid:
    def test_valid_single_key(self):
        parsed = _make_parsed({
            "tool_call": "communication_rules",
            "keys": ["GM-RULE-COMMUNICATION-SOFTENING"],
        }, is_tool_call=True)
        result = check_communication_rules_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_valid_multiple_keys(self):
        parsed = _make_parsed({
            "tool_call": "communication_rules",
            "keys": ["GM-RULE-COMMUNICATION-SOFTENING", "GM-RULE-EVASION-DEFINITION"],
        }, is_tool_call=True)
        result = check_communication_rules_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_string_key_accepted(self):
        parsed = _make_parsed({
            "tool_call": "communication_rules",
            "keys": "GM-RULE-COMMUNICATION-ACTION",
        }, is_tool_call=True)
        result = check_communication_rules_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_missing_keys(self):
        parsed = _make_parsed({
            "tool_call": "communication_rules",
        }, is_tool_call=True)
        result = check_communication_rules_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed
        assert "Missing 'keys'" in result.detail

    def test_invalid_key(self):
        parsed = _make_parsed({
            "tool_call": "communication_rules",
            "keys": ["INVALID-KEY"],
        }, is_tool_call=True)
        result = check_communication_rules_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed
        assert "not a valid" in result.detail

    def test_too_many_keys(self):
        parsed = _make_parsed({
            "tool_call": "communication_rules",
            "keys": [f"GM-RULE-COMMUNICATION-SOFTENING" for _ in range(9)],
        }, is_tool_call=True)
        result = check_communication_rules_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed
        assert "Too many keys" in result.detail

    def test_case_insensitive_key(self):
        """Keys are uppercased before matching."""
        parsed = _make_parsed({
            "tool_call": "communication_rules",
            "keys": ["gm-rule-communication-softening"],
        }, is_tool_call=True)
        result = check_communication_rules_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_skipped_for_other_tools(self):
        parsed = _make_parsed({
            "tool_call": "memory_search",
            "queries": ["test"],
        }, is_tool_call=True)
        result = check_communication_rules_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed


# ── name_generate_valid ─────────────────────────────────────────


class TestNameGenerateValid:
    def test_valid_minimal(self):
        parsed = _make_parsed({
            "tool_call": "name_generate",
        }, is_tool_call=True)
        result = check_name_generate_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_valid_full(self):
        parsed = _make_parsed({
            "tool_call": "name_generate",
            "origins": ["italian", "arabic"],
            "gender": "f",
            "count": 5,
            "context": "confident bartender in her 40s",
        }, is_tool_call=True)
        result = check_name_generate_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_origins_as_string(self):
        parsed = _make_parsed({
            "tool_call": "name_generate",
            "origins": "italian",
        }, is_tool_call=True)
        result = check_name_generate_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_too_many_origins(self):
        parsed = _make_parsed({
            "tool_call": "name_generate",
            "origins": ["a", "b", "c", "d", "e"],
        }, is_tool_call=True)
        result = check_name_generate_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed
        assert "Too many origins" in result.detail

    def test_invalid_gender(self):
        parsed = _make_parsed({
            "tool_call": "name_generate",
            "gender": "unknown",
        }, is_tool_call=True)
        result = check_name_generate_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed
        assert "gender" in result.detail

    def test_count_out_of_range(self):
        parsed = _make_parsed({
            "tool_call": "name_generate",
            "count": 10,
        }, is_tool_call=True)
        result = check_name_generate_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed
        assert "count" in result.detail

    def test_count_zero(self):
        parsed = _make_parsed({
            "tool_call": "name_generate",
            "count": 0,
        }, is_tool_call=True)
        result = check_name_generate_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed

    def test_context_too_long(self):
        parsed = _make_parsed({
            "tool_call": "name_generate",
            "context": "x" * 301,
        }, is_tool_call=True)
        result = check_name_generate_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert not result.passed
        assert "context" in result.detail

    def test_valid_gender_m(self):
        parsed = _make_parsed({
            "tool_call": "name_generate",
            "gender": "m",
        }, is_tool_call=True)
        result = check_name_generate_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_valid_gender_both(self):
        parsed = _make_parsed({
            "tool_call": "name_generate",
            "gender": "both",
        }, is_tool_call=True)
        result = check_name_generate_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_skipped_for_other_tools(self):
        parsed = _make_parsed({
            "tool_call": "memory_search",
            "queries": ["test"],
        }, is_tool_call=True)
        result = check_name_generate_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed


# ── Chapter close action ────────────────────────────────────────


class TestChapterCloseAction:
    def test_close_action_valid(self):
        parsed = _make_parsed({
            "tool_call": "chapter_plan",
            "action": "close",
            "chapter": "chapter-1",
        }, is_tool_call=True)
        result = chapter_fields_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_close_action_with_dict(self):
        parsed = _make_parsed({
            "tool_call": "chapter_plan",
            "action": "close",
            "chapter": {"slug": "chapter-1"},
        }, is_tool_call=True)
        result = chapter_fields_valid(parsed, _make_scenario(), TURN, _make_state(), P)
        assert result.passed

    def test_close_updates_state(self):
        """Close action sets chapter status to resolved in AccumulatedState."""
        state = _make_state()
        # Create a chapter first
        state.apply({
            "tool_call": "chapter_plan",
            "action": "create",
            "chapter": {"slug": "intro", "title": "Introduction", "scenes": ["s1", "s2"]},
        })
        assert "intro" in state.chapters
        assert state.chapters["intro"]["status"] == "active"

        # Close it
        state.apply({
            "tool_call": "chapter_plan",
            "action": "close",
            "chapter": "intro",
            "resolution": "The hero departed.",
        })
        assert state.chapters["intro"]["status"] == "resolved"
        assert state.chapters["intro"]["resolution"] == "The hero departed."

    def test_close_with_dict_updates_state(self):
        state = _make_state()
        state.apply({
            "tool_call": "chapter_plan",
            "action": "create",
            "chapter": {"slug": "act-2", "title": "Act 2", "scenes": ["s1"]},
        })
        state.apply({
            "tool_call": "chapter_plan",
            "action": "close",
            "chapter": {"slug": "act-2"},
            "resolution": "Conflict resolved.",
        })
        assert state.chapters["act-2"]["status"] == "resolved"
        assert state.chapters["act-2"]["resolution"] == "Conflict resolved."

    def test_close_resolution_truncated(self):
        """Resolution is truncated to 260 chars like the engine does."""
        state = _make_state()
        state.apply({
            "tool_call": "chapter_plan",
            "action": "create",
            "chapter": {"slug": "ch-1", "title": "Chapter 1", "scenes": ["s1"]},
        })
        state.apply({
            "tool_call": "chapter_plan",
            "action": "close",
            "chapter": "ch-1",
            "resolution": "x" * 300,
        })
        assert len(state.chapters["ch-1"]["resolution"]) == 260


# ── Registry check: new checks registered ──────────────────────


class TestRegistryRound5:
    def test_communication_rules_valid_registered(self):
        from tgb.checks.registry import get_check
        fn = get_check("communication_rules_valid")
        assert fn is check_communication_rules_valid

    def test_name_generate_valid_registered(self):
        from tgb.checks.registry import get_check
        fn = get_check("name_generate_valid")
        assert fn is check_name_generate_valid
