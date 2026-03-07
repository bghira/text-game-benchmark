"""Tests for subplot checks — plot threads, chapters, consequences."""

import pytest

from tgb.checks.subplot import (
    plot_thread_fields_valid,
    plot_thread_target_reasonable,
    plot_thread_not_orphaned,
    chapter_fields_valid,
    chapter_scene_progression,
    consequence_fields_valid,
    consequence_severity_proportional,
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
        turns=kwargs.get("turns", [TurnSpec(action="test")]),
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
    if not raw and json_data:
        import json
        raw = json.dumps(json_data)
    return ParsedResponse(
        raw=raw,
        parsed_json=json_data,
        parse_error="",
        is_tool_call=False,
    )


TURN = TurnSpec(action="test")
EMPTY_PARAMS: dict = {}


# ── plot_thread_fields_valid ─────────────────────────────────────────


class TestPlotThreadFieldsValid:
    def test_skip_non_plot_plan(self):
        parsed = _make_parsed({"tool_call": "memory", "query": "test"})
        result = plot_thread_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed
        assert "skipped" in result.detail

    def test_pass_valid_plan(self):
        parsed = _make_parsed({
            "tool_call": "plot_plan",
            "plans": [{
                "thread": "rescue-the-queen",
                "target_turns": 5,
                "setup": "The queen has been captured by the Jabberwock.",
                "intended_payoff": "Alice frees the queen.",
                "status": "active",
            }],
        })
        result = plot_thread_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed
        assert "1 thread" in result.detail

    def test_fail_empty_plans(self):
        parsed = _make_parsed({"tool_call": "plot_plan", "plans": []})
        result = plot_thread_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "empty" in result.detail

    def test_fail_missing_plans(self):
        parsed = _make_parsed({"tool_call": "plot_plan"})
        result = plot_thread_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed

    def test_fail_bad_slug(self):
        parsed = _make_parsed({
            "tool_call": "plot_plan",
            "plans": [{"thread": "Not A Slug", "target_turns": 3}],
        })
        result = plot_thread_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "kebab-case" in result.detail

    def test_fail_missing_thread(self):
        parsed = _make_parsed({
            "tool_call": "plot_plan",
            "plans": [{"target_turns": 3}],
        })
        result = plot_thread_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "missing" in result.detail

    def test_fail_target_turns_out_of_range(self):
        parsed = _make_parsed({
            "tool_call": "plot_plan",
            "plans": [{"thread": "test-thread", "target_turns": 300}],
        })
        result = plot_thread_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "out of range" in result.detail

    def test_fail_target_turns_zero(self):
        parsed = _make_parsed({
            "tool_call": "plot_plan",
            "plans": [{"thread": "test-thread", "target_turns": 0}],
        })
        result = plot_thread_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed

    def test_fail_target_turns_not_numeric(self):
        parsed = _make_parsed({
            "tool_call": "plot_plan",
            "plans": [{"thread": "test-thread", "target_turns": "five"}],
        })
        result = plot_thread_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "not numeric" in result.detail

    def test_fail_bad_status(self):
        parsed = _make_parsed({
            "tool_call": "plot_plan",
            "plans": [{"thread": "test-thread", "status": "paused"}],
        })
        result = plot_thread_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "status" in result.detail

    def test_fail_field_too_long(self):
        parsed = _make_parsed({
            "tool_call": "plot_plan",
            "plans": [{"thread": "test-thread", "setup": "x" * 300}],
        })
        result = plot_thread_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "exceeds" in result.detail

    def test_fail_dependencies_too_many(self):
        parsed = _make_parsed({
            "tool_call": "plot_plan",
            "plans": [{"thread": "test-thread", "dependencies": list(range(10))}],
        })
        result = plot_thread_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "dependencies" in result.detail

    def test_fail_dependencies_not_list(self):
        parsed = _make_parsed({
            "tool_call": "plot_plan",
            "plans": [{"thread": "test-thread", "dependencies": "not-a-list"}],
        })
        result = plot_thread_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed

    def test_pass_multiple_valid_plans(self):
        parsed = _make_parsed({
            "tool_call": "plot_plan",
            "plans": [
                {"thread": "thread-a", "target_turns": 3, "status": "active"},
                {"thread": "thread-b", "target_turns": 10, "status": "active"},
            ],
        })
        result = plot_thread_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed
        assert "2 thread" in result.detail

    def test_fail_plan_not_dict(self):
        parsed = _make_parsed({
            "tool_call": "plot_plan",
            "plans": ["not a dict"],
        })
        result = plot_thread_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "not a dict" in result.detail


# ── plot_thread_target_reasonable ────────────────────────────────────


class TestPlotThreadTargetReasonable:
    def test_skip_non_plot_plan(self):
        parsed = _make_parsed({"narration": "test"})
        result = plot_thread_target_reasonable(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_pass_reasonable_target(self):
        parsed = _make_parsed({
            "tool_call": "plot_plan",
            "plans": [{"thread": "test-thread", "target_turns": 5, "status": "active"}],
        })
        scenario = _make_scenario(turns=[TurnSpec(action=f"t{i}") for i in range(10)])
        result = plot_thread_target_reasonable(parsed, scenario, TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_fail_target_turns_one(self):
        parsed = _make_parsed({
            "tool_call": "plot_plan",
            "plans": [{"thread": "tiny-thread", "target_turns": 1}],
        })
        result = plot_thread_target_reasonable(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "single-turn" in result.detail

    def test_fail_target_wildly_large(self):
        parsed = _make_parsed({
            "tool_call": "plot_plan",
            "plans": [{"thread": "huge-thread", "target_turns": 200, "status": "active"}],
        })
        scenario = _make_scenario(turns=[TurnSpec(action="t1"), TurnSpec(action="t2")])
        result = plot_thread_target_reasonable(parsed, scenario, TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "2-turn scenario" in result.detail

    def test_resolved_thread_skips_scenario_check(self):
        parsed = _make_parsed({
            "tool_call": "plot_plan",
            "plans": [{"thread": "done-thread", "target_turns": 200, "status": "resolved"}],
        })
        scenario = _make_scenario(turns=[TurnSpec(action="t1")])
        result = plot_thread_target_reasonable(parsed, scenario, TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_skip_no_target(self):
        parsed = _make_parsed({
            "tool_call": "plot_plan",
            "plans": [{"thread": "no-target"}],
        })
        result = plot_thread_target_reasonable(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed


# ── plot_thread_not_orphaned ─────────────────────────────────────────


class TestPlotThreadNotOrphaned:
    def test_pass_no_threads(self):
        state = _make_state()
        parsed = _make_parsed({"narration": "test"})
        result = plot_thread_not_orphaned(parsed, _make_scenario(), TURN, state, EMPTY_PARAMS)
        assert result.passed
        assert "0 thread" in result.detail

    def test_pass_thread_within_deadline(self):
        state = _make_state()
        state.plot_threads = {
            "rescue-mission": {
                "thread": "rescue-mission",
                "target_turns": 5,
                "created_turn": 1,
                "status": "active",
            },
        }
        state.turn_number = 4  # Still within deadline (1 + 5 = 6)
        parsed = _make_parsed({"narration": "test"})
        result = plot_thread_not_orphaned(parsed, _make_scenario(), TURN, state, EMPTY_PARAMS)
        assert result.passed

    def test_fail_thread_overdue(self):
        state = _make_state()
        state.plot_threads = {
            "rescue-mission": {
                "thread": "rescue-mission",
                "target_turns": 3,
                "created_turn": 1,
                "status": "active",
            },
        }
        state.turn_number = 10  # Well past deadline (1 + 3 = 4)
        parsed = _make_parsed({"narration": "test"})
        result = plot_thread_not_orphaned(parsed, _make_scenario(), TURN, state, EMPTY_PARAMS)
        assert not result.passed
        assert "overdue" in result.detail

    def test_pass_resolved_thread_ignored(self):
        state = _make_state()
        state.plot_threads = {
            "old-thread": {
                "thread": "old-thread",
                "target_turns": 2,
                "created_turn": 1,
                "status": "resolved",
            },
        }
        state.turn_number = 100
        parsed = _make_parsed({"narration": "test"})
        result = plot_thread_not_orphaned(parsed, _make_scenario(), TURN, state, EMPTY_PARAMS)
        assert result.passed

    def test_fail_multiple_overdue(self):
        state = _make_state()
        state.plot_threads = {
            "thread-a": {"target_turns": 2, "created_turn": 1, "status": "active"},
            "thread-b": {"target_turns": 3, "created_turn": 1, "status": "active"},
        }
        state.turn_number = 20
        parsed = _make_parsed({"narration": "test"})
        result = plot_thread_not_orphaned(parsed, _make_scenario(), TURN, state, EMPTY_PARAMS)
        assert not result.passed
        assert "thread-a" in result.detail
        assert "thread-b" in result.detail

    def test_pass_thread_no_target(self):
        """Thread without target_turns is not checked for overdue."""
        state = _make_state()
        state.plot_threads = {
            "open-ended": {"thread": "open-ended", "status": "active", "created_turn": 1},
        }
        state.turn_number = 100
        parsed = _make_parsed({"narration": "test"})
        result = plot_thread_not_orphaned(parsed, _make_scenario(), TURN, state, EMPTY_PARAMS)
        assert result.passed


# ── chapter_fields_valid ─────────────────────────────────────────────


class TestChapterFieldsValid:
    def test_skip_non_chapter_plan(self):
        parsed = _make_parsed({"tool_call": "plot_plan", "plans": []})
        result = chapter_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed
        assert "skipped" in result.detail

    def test_pass_create(self):
        parsed = _make_parsed({
            "tool_call": "chapter_plan",
            "action": "create",
            "chapter": {
                "slug": "the-chase",
                "title": "The Chase",
                "scenes": ["pursuit", "cornered", "escape"],
            },
        })
        result = chapter_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed
        assert "create" in result.detail

    def test_fail_invalid_action(self):
        parsed = _make_parsed({
            "tool_call": "chapter_plan",
            "action": "delete",
        })
        result = chapter_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "not valid" in result.detail

    def test_fail_create_bad_slug(self):
        parsed = _make_parsed({
            "tool_call": "chapter_plan",
            "action": "create",
            "chapter": {
                "slug": "Bad Slug!",
                "scenes": ["a", "b"],
            },
        })
        result = chapter_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "slug" in result.detail.lower()

    def test_fail_create_empty_scenes(self):
        parsed = _make_parsed({
            "tool_call": "chapter_plan",
            "action": "create",
            "chapter": {
                "slug": "test-chapter",
                "scenes": [],
            },
        })
        result = chapter_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "scenes" in result.detail.lower()

    def test_fail_create_too_many_scenes(self):
        parsed = _make_parsed({
            "tool_call": "chapter_plan",
            "action": "create",
            "chapter": {
                "slug": "long-chapter",
                "scenes": [f"scene-{i}" for i in range(25)],
            },
        })
        result = chapter_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "exceeds 20" in result.detail

    def test_fail_create_title_too_long(self):
        parsed = _make_parsed({
            "tool_call": "chapter_plan",
            "action": "create",
            "chapter": {
                "slug": "test-chapter",
                "title": "x" * 150,
                "scenes": ["a"],
            },
        })
        result = chapter_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "120" in result.detail

    def test_pass_advance_scene_string_slug(self):
        parsed = _make_parsed({
            "tool_call": "chapter_plan",
            "action": "advance_scene",
            "chapter": "the-chase",
            "to_scene": "cornered",
        })
        result = chapter_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_fail_advance_scene_bad_slug(self):
        parsed = _make_parsed({
            "tool_call": "chapter_plan",
            "action": "advance_scene",
            "chapter": "BAD SLUG",
        })
        result = chapter_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed

    def test_pass_resolve_dict_slug(self):
        parsed = _make_parsed({
            "tool_call": "chapter_plan",
            "action": "resolve",
            "chapter": {"slug": "the-chase"},
        })
        result = chapter_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_fail_resolve_dict_missing_slug(self):
        parsed = _make_parsed({
            "tool_call": "chapter_plan",
            "action": "resolve",
            "chapter": {"title": "oops"},
        })
        result = chapter_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed


# ── chapter_scene_progression ────────────────────────────────────────


class TestChapterSceneProgression:
    def test_skip_non_chapter(self):
        parsed = _make_parsed({"tool_call": "plot_plan"})
        result = chapter_scene_progression(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_skip_non_advance(self):
        parsed = _make_parsed({"tool_call": "chapter_plan", "action": "create"})
        result = chapter_scene_progression(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_pass_correct_advancement(self):
        state = _make_state()
        state.chapters = {
            "the-chase": {
                "slug": "the-chase",
                "scenes": ["pursuit", "cornered", "escape"],
                "current_scene": "pursuit",
            },
        }
        parsed = _make_parsed({
            "tool_call": "chapter_plan",
            "action": "advance_scene",
            "chapter": "the-chase",
            "to_scene": "cornered",
        })
        result = chapter_scene_progression(parsed, _make_scenario(), TURN, state, EMPTY_PARAMS)
        assert result.passed
        assert "correctly" in result.detail

    def test_fail_skipped_scene(self):
        state = _make_state()
        state.chapters = {
            "the-chase": {
                "slug": "the-chase",
                "scenes": ["pursuit", "cornered", "escape"],
                "current_scene": "pursuit",
            },
        }
        parsed = _make_parsed({
            "tool_call": "chapter_plan",
            "action": "advance_scene",
            "chapter": "the-chase",
            "to_scene": "escape",
        })
        result = chapter_scene_progression(parsed, _make_scenario(), TURN, state, EMPTY_PARAMS)
        assert not result.passed
        assert "Skipped" in result.detail

    def test_fail_backward_scene(self):
        state = _make_state()
        state.chapters = {
            "the-chase": {
                "slug": "the-chase",
                "scenes": ["pursuit", "cornered", "escape"],
                "current_scene": "cornered",
            },
        }
        parsed = _make_parsed({
            "tool_call": "chapter_plan",
            "action": "advance_scene",
            "chapter": "the-chase",
            "to_scene": "pursuit",
        })
        result = chapter_scene_progression(parsed, _make_scenario(), TURN, state, EMPTY_PARAMS)
        assert not result.passed
        assert "backward" in result.detail

    def test_skip_untracked_chapter(self):
        state = _make_state()
        parsed = _make_parsed({
            "tool_call": "chapter_plan",
            "action": "advance_scene",
            "chapter": "unknown-chapter",
            "to_scene": "scene-1",
        })
        result = chapter_scene_progression(parsed, _make_scenario(), TURN, state, EMPTY_PARAMS)
        assert result.passed
        assert "not tracked" in result.detail

    def test_pass_dict_chapter_ref(self):
        state = _make_state()
        state.chapters = {
            "the-chase": {
                "slug": "the-chase",
                "scenes": ["a", "b", "c"],
                "current_scene": "a",
            },
        }
        parsed = _make_parsed({
            "tool_call": "chapter_plan",
            "action": "advance_scene",
            "chapter": {"slug": "the-chase"},
            "to_scene": "b",
        })
        result = chapter_scene_progression(parsed, _make_scenario(), TURN, state, EMPTY_PARAMS)
        assert result.passed

    def test_skip_insufficient_data(self):
        state = _make_state()
        state.chapters = {
            "empty-chapter": {"slug": "empty-chapter", "scenes": [], "current_scene": ""},
        }
        parsed = _make_parsed({
            "tool_call": "chapter_plan",
            "action": "advance_scene",
            "chapter": "empty-chapter",
            "to_scene": "a",
        })
        result = chapter_scene_progression(parsed, _make_scenario(), TURN, state, EMPTY_PARAMS)
        assert result.passed
        assert "Insufficient" in result.detail


# ── consequence_fields_valid ─────────────────────────────────────────


class TestConsequenceFieldsValid:
    def test_skip_non_consequence(self):
        parsed = _make_parsed({"tool_call": "plot_plan"})
        result = consequence_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_pass_valid_add(self):
        parsed = _make_parsed({
            "tool_call": "consequence_log",
            "add": [{
                "trigger": "guards alerted",
                "consequence": "patrols increase in the area",
                "severity": "moderate",
                "expires_turns": 5,
            }],
        })
        result = consequence_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_pass_single_dict_add(self):
        """add can be a single dict instead of a list."""
        parsed = _make_parsed({
            "tool_call": "consequence_log",
            "add": {
                "trigger": "door broken",
                "consequence": "guards will investigate the noise",
                "severity": "low",
            },
        })
        result = consequence_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_fail_missing_trigger(self):
        parsed = _make_parsed({
            "tool_call": "consequence_log",
            "add": [{"consequence": "something happens"}],
        })
        result = consequence_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "trigger" in result.detail

    def test_fail_missing_consequence(self):
        parsed = _make_parsed({
            "tool_call": "consequence_log",
            "add": [{"trigger": "something"}],
        })
        result = consequence_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "consequence" in result.detail

    def test_fail_invalid_severity(self):
        parsed = _make_parsed({
            "tool_call": "consequence_log",
            "add": [{"trigger": "t", "consequence": "c", "severity": "extreme"}],
        })
        result = consequence_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "severity" in result.detail

    def test_fail_negative_expires(self):
        parsed = _make_parsed({
            "tool_call": "consequence_log",
            "add": [{"trigger": "t", "consequence": "c", "expires_turns": -1}],
        })
        result = consequence_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "negative" in result.detail

    def test_fail_expires_not_numeric(self):
        parsed = _make_parsed({
            "tool_call": "consequence_log",
            "add": [{"trigger": "t", "consequence": "c", "expires_turns": "forever"}],
        })
        result = consequence_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "not numeric" in result.detail

    def test_fail_trigger_too_long(self):
        parsed = _make_parsed({
            "tool_call": "consequence_log",
            "add": [{"trigger": "x" * 250, "consequence": "c"}],
        })
        result = consequence_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "240" in result.detail

    def test_fail_consequence_too_long(self):
        parsed = _make_parsed({
            "tool_call": "consequence_log",
            "add": [{"trigger": "t", "consequence": "x" * 350}],
        })
        result = consequence_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "300" in result.detail

    def test_pass_valid_resolve(self):
        parsed = _make_parsed({
            "tool_call": "consequence_log",
            "resolve": [{"id": "guards-alerted", "resolution": "guards stood down"}],
        })
        result = consequence_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_fail_resolve_missing_id(self):
        parsed = _make_parsed({
            "tool_call": "consequence_log",
            "resolve": [{"resolution": "resolved somehow"}],
        })
        result = consequence_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "id" in result.detail

    def test_pass_single_dict_resolve(self):
        """resolve can be a single dict instead of a list."""
        parsed = _make_parsed({
            "tool_call": "consequence_log",
            "resolve": {"id": "test-id"},
        })
        result = consequence_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_pass_valid_severities(self):
        for sev in ("low", "moderate", "high", "critical"):
            parsed = _make_parsed({
                "tool_call": "consequence_log",
                "add": [{"trigger": "t", "consequence": "c", "severity": sev}],
            })
            result = consequence_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
            assert result.passed, f"severity '{sev}' should be valid"

    def test_fail_add_not_dict(self):
        parsed = _make_parsed({
            "tool_call": "consequence_log",
            "add": ["not a dict"],
        })
        result = consequence_fields_valid(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "not a dict" in result.detail


# ── consequence_severity_proportional ────────────────────────────────


class TestConsequenceSeverityProportional:
    def test_skip_non_consequence(self):
        parsed = _make_parsed({"tool_call": "plot_plan"})
        result = consequence_severity_proportional(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_pass_single_critical(self):
        parsed = _make_parsed({
            "tool_call": "consequence_log",
            "add": [{"trigger": "t", "consequence": "c", "severity": "critical"}],
        })
        result = consequence_severity_proportional(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_fail_multiple_critical(self):
        parsed = _make_parsed({
            "tool_call": "consequence_log",
            "add": [
                {"trigger": "t1", "consequence": "c1", "severity": "critical"},
                {"trigger": "t2", "consequence": "c2", "severity": "critical"},
            ],
        })
        result = consequence_severity_proportional(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed
        assert "disproportionate" in result.detail

    def test_fail_too_many_high_and_critical(self):
        parsed = _make_parsed({
            "tool_call": "consequence_log",
            "add": [
                {"trigger": "t1", "consequence": "c1", "severity": "critical"},
                {"trigger": "t2", "consequence": "c2", "severity": "high"},
                {"trigger": "t3", "consequence": "c3", "severity": "high"},
            ],
        })
        result = consequence_severity_proportional(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert not result.passed

    def test_pass_many_low_severity(self):
        parsed = _make_parsed({
            "tool_call": "consequence_log",
            "add": [
                {"trigger": f"t{i}", "consequence": f"c{i}", "severity": "low"}
                for i in range(5)
            ],
        })
        result = consequence_severity_proportional(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_pass_two_high_no_critical(self):
        parsed = _make_parsed({
            "tool_call": "consequence_log",
            "add": [
                {"trigger": "t1", "consequence": "c1", "severity": "high"},
                {"trigger": "t2", "consequence": "c2", "severity": "high"},
            ],
        })
        result = consequence_severity_proportional(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_pass_no_add(self):
        parsed = _make_parsed({
            "tool_call": "consequence_log",
            "resolve": [{"id": "test"}],
        })
        result = consequence_severity_proportional(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed

    def test_pass_single_dict_add(self):
        parsed = _make_parsed({
            "tool_call": "consequence_log",
            "add": {"trigger": "t", "consequence": "c", "severity": "high"},
        })
        result = consequence_severity_proportional(parsed, _make_scenario(), TURN, _make_state(), EMPTY_PARAMS)
        assert result.passed


# ── Registry integration ─────────────────────────────────────────────


class TestSubplotRegistry:
    def test_all_subplot_checks_in_registry(self):
        from tgb.checks.registry import CHECKS
        subplot_ids = [
            "plot_thread_fields_valid",
            "plot_thread_target_reasonable",
            "plot_thread_not_orphaned",
            "chapter_fields_valid",
            "chapter_scene_progression",
            "consequence_fields_valid",
            "consequence_severity_proportional",
        ]
        for check_id in subplot_ids:
            assert check_id in CHECKS, f"'{check_id}' missing from registry"

    def test_registry_count_includes_subplot(self):
        from tgb.checks.registry import list_checks
        checks = list_checks()
        assert len(checks) >= 34  # 27 prior + 7 subplot


# ── AccumulatedState subplot tracking ────────────────────────────────


class TestAccumulatedStateSubplot:
    def test_apply_plot_plan(self):
        state = _make_state()
        state.apply({
            "tool_call": "plot_plan",
            "plans": [
                {"thread": "rescue-quest", "target_turns": 5, "setup": "hero is captured"},
            ],
        })
        assert "rescue-quest" in state.plot_threads
        assert state.plot_threads["rescue-quest"]["created_turn"] == 1
        assert state.plot_threads["rescue-quest"]["target_turns"] == 5

    def test_apply_plot_plan_update(self):
        state = _make_state()
        state.apply({
            "tool_call": "plot_plan",
            "plans": [{"thread": "quest", "target_turns": 5, "status": "active"}],
        })
        state.apply({
            "tool_call": "plot_plan",
            "plans": [{"thread": "quest", "status": "resolved", "resolution": "done"}],
        })
        assert state.plot_threads["quest"]["status"] == "resolved"
        assert state.plot_threads["quest"]["created_turn"] == 1
        assert state.plot_threads["quest"]["updated_turn"] == 2

    def test_apply_chapter_create(self):
        state = _make_state()
        state.apply({
            "tool_call": "chapter_plan",
            "action": "create",
            "chapter": {
                "slug": "the-chase",
                "scenes": ["pursuit", "cornered", "escape"],
            },
        })
        assert "the-chase" in state.chapters
        assert state.chapters["the-chase"]["scenes"] == ["pursuit", "cornered", "escape"]
        assert state.chapters["the-chase"]["status"] == "active"

    def test_apply_chapter_advance_scene(self):
        state = _make_state()
        state.apply({
            "tool_call": "chapter_plan",
            "action": "create",
            "chapter": {"slug": "ch1", "scenes": ["a", "b", "c"]},
        })
        state.apply({
            "tool_call": "chapter_plan",
            "action": "advance_scene",
            "chapter": "ch1",
            "to_scene": "b",
        })
        assert state.chapters["ch1"]["current_scene"] == "b"

    def test_apply_chapter_resolve(self):
        state = _make_state()
        state.apply({
            "tool_call": "chapter_plan",
            "action": "create",
            "chapter": {"slug": "ch1", "scenes": ["a"]},
        })
        state.apply({
            "tool_call": "chapter_plan",
            "action": "resolve",
            "chapter": "ch1",
            "resolution": "chapter complete",
        })
        assert state.chapters["ch1"]["status"] == "resolved"

    def test_apply_consequence_add(self):
        state = _make_state()
        state.apply({
            "tool_call": "consequence_log",
            "add": [{
                "id": "guards-alert",
                "trigger": "broke the door",
                "consequence": "guards patrolling",
                "expires_turns": 3,
            }],
        })
        assert "guards-alert" in state.consequences
        assert state.consequences["guards-alert"]["expires_at_turn"] == 4  # turn 1 + 3

    def test_apply_consequence_resolve(self):
        state = _make_state()
        state.apply({
            "tool_call": "consequence_log",
            "add": [{"id": "alert", "trigger": "t", "consequence": "c"}],
        })
        state.apply({
            "tool_call": "consequence_log",
            "resolve": [{"id": "alert", "resolution": "guards left"}],
        })
        assert state.consequences["alert"]["status"] == "resolved"

    def test_apply_consequence_remove(self):
        state = _make_state()
        state.apply({
            "tool_call": "consequence_log",
            "add": [{"id": "temp", "trigger": "t", "consequence": "c"}],
        })
        assert "temp" in state.consequences
        state.apply({
            "tool_call": "consequence_log",
            "remove": ["temp"],
        })
        assert "temp" not in state.consequences

    def test_tool_call_history_tracked(self):
        state = _make_state()
        state.apply({
            "tool_call": "plot_plan",
            "plans": [{"thread": "a", "target_turns": 3}],
        })
        state.apply({
            "tool_call": "consequence_log",
            "add": [{"trigger": "t", "consequence": "c"}],
        })
        assert len(state.tool_call_history) == 2
        assert state.tool_call_history[0]["tool_call"] == "plot_plan"
        assert state.tool_call_history[1]["tool_call"] == "consequence_log"
