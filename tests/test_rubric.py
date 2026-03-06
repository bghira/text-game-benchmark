"""Tests for tgb.rubric — rubric loading, grading, and scoring."""

import json
import tempfile
from pathlib import Path

import pytest

from tgb.config import Scenario, CampaignSetup, PlayerSetup, TurnSpec, CheckSpec
from tgb.prompt_builder import AccumulatedState
from tgb.response_parser import ParsedResponse
from tgb.rubric import (
    Rubric,
    RubricLevel,
    RubricScore,
    RubricGrader,
    _parse_rubric,
    load_rubrics,
    load_rubrics_from_file,
    builtin_rubric_dir,
    RUBRIC_JUDGE_SYSTEM_PROMPT,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_scenario(n_turns=1):
    turns = [
        TurnSpec(action=f"action {i}", action_id=f"turn-{i}", checks=[])
        for i in range(n_turns)
    ]
    return Scenario(
        name="test",
        description="test",
        tags=[],
        tier="basic",
        campaign=CampaignSetup(
            name="test",
            state={"tone": "dark fantasy"},
            characters={
                "wizard": {"name": "Gandalf", "personality": "wise"}
            },
        ),
        player=PlayerSetup(),
        turns=turns,
    )


def _make_parsed(narration="You enter a dark room.", **extra):
    data = {"narration": narration, "reasoning": "test", **extra}
    return ParsedResponse(raw=json.dumps(data), parsed_json=data)


def _make_rubric(
    rubric_id="test-rubric",
    scope="turn",
    computed_metric="",
    category="craft",
):
    return Rubric(
        id=rubric_id,
        name=f"Test Rubric ({rubric_id})",
        category=category,
        description="A test rubric",
        levels=[
            RubricLevel(score=5, label="", description="Excellent"),
            RubricLevel(score=4, label="", description="Good"),
            RubricLevel(score=3, label="", description="Acceptable"),
            RubricLevel(score=2, label="", description="Poor"),
            RubricLevel(score=1, label="", description="Bad"),
        ],
        scope=scope,
        computed_metric=computed_metric,
    )


class FakeRubricJudgeClient:
    """Fake client that returns predefined rubric grades."""

    def __init__(self, grades: list[dict]):
        self.grades = grades
        self.calls: list[dict] = []

    def complete_text(self, system_prompt, user_prompt, **kwargs):
        self.calls.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            **kwargs,
        })
        response = json.dumps({"grades": self.grades})
        from tgb.clients.ollama_client import TimingData
        return response, TimingData()


# ── _parse_rubric ────────────────────────────────────────────────────────

class TestParseRubric:
    def test_simple_format(self):
        raw = {
            "id": "voice",
            "name": "Voice",
            "category": "craft",
            "description": "Test voice rubric",
            "scope": "turn",
            "levels": {5: "Great", 4: "Good", 3: "OK", 2: "Poor", 1: "Bad"},
        }
        rubric = _parse_rubric(raw)
        assert rubric.id == "voice"
        assert rubric.name == "Voice"
        assert len(rubric.levels) == 5
        assert rubric.levels[0].score == 5  # sorted descending
        assert rubric.levels[4].score == 1

    def test_missing_id(self):
        with pytest.raises(ValueError, match="missing 'id'"):
            _parse_rubric({"name": "no id"})

    def test_missing_levels(self):
        with pytest.raises(ValueError, match="missing or empty 'levels'"):
            _parse_rubric({"id": "x", "levels": {}})

    def test_computed_metric(self):
        raw = {
            "id": "rep",
            "name": "Rep",
            "category": "craft",
            "description": "Rep",
            "scope": "scenario",
            "computed_metric": "cosine_similarity",
            "levels": {5: "Varied", 1: "Repetitive"},
        }
        rubric = _parse_rubric(raw)
        assert rubric.computed_metric == "cosine_similarity"
        assert rubric.scope == "scenario"

    def test_max_score(self):
        rubric = _make_rubric()
        assert rubric.max_score == 5


# ── load_rubrics ─────────────────────────────────────────────────────────

class TestLoadRubrics:
    def test_load_from_file(self, tmp_path):
        rubric_yaml = tmp_path / "test.yaml"
        rubric_yaml.write_text(
            "id: my-rubric\n"
            "name: My Rubric\n"
            "category: craft\n"
            "description: Test\n"
            "levels:\n"
            "  5: Great\n"
            "  3: OK\n"
            "  1: Bad\n"
        )
        rubrics = load_rubrics_from_file(rubric_yaml)
        assert len(rubrics) == 1
        assert rubrics[0].id == "my-rubric"

    def test_load_from_dir(self, tmp_path):
        for i in range(3):
            f = tmp_path / f"rubric{i}.yaml"
            f.write_text(
                f"id: r{i}\nname: R{i}\ncategory: craft\n"
                f"description: test\nlevels:\n  5: Good\n  1: Bad\n"
            )
        # Also add a _hidden file that should be skipped
        (tmp_path / "_hidden.yaml").write_text(
            "id: hidden\nname: H\ncategory: x\ndescription: x\nlevels:\n  5: x\n  1: x\n"
        )
        rubrics = load_rubrics([tmp_path])
        assert len(rubrics) == 3
        assert "hidden" not in rubrics

    def test_nonexistent_dir(self):
        rubrics = load_rubrics(["/nonexistent/path"])
        assert rubrics == {}

    def test_builtin_rubrics_exist(self):
        d = builtin_rubric_dir()
        assert d.exists(), f"Built-in rubrics dir not found: {d}"
        rubrics = load_rubrics([d])
        assert len(rubrics) >= 8
        assert "voice" in rubrics
        assert "pacing" in rubrics
        assert "consent" in rubrics
        assert "proportionality" in rubrics
        assert "contrivance" in rubrics
        assert "repetitiveness" in rubrics
        assert "causality" in rubrics
        assert "npc-autonomy" in rubrics

    def test_builtin_repetitiveness_has_metric(self):
        rubrics = load_rubrics([builtin_rubric_dir()])
        rep = rubrics["repetitiveness"]
        assert rep.computed_metric == "cosine_similarity"
        assert rep.scope == "scenario"


# ── RubricScore ──────────────────────────────────────────────────────────

class TestRubricScore:
    def test_to_dict_basic(self):
        rs = RubricScore(
            rubric_id="voice",
            rubric_name="Voice",
            category="craft",
            score=4,
            max_score=5,
            reason="Good voice",
            scope="turn",
            action_id="turn-0",
        )
        d = rs.to_dict()
        assert d["rubric_id"] == "voice"
        assert d["score"] == 4
        assert d["action_id"] == "turn-0"
        assert "metric_name" not in d

    def test_to_dict_with_metric(self):
        rs = RubricScore(
            rubric_id="repetitiveness",
            rubric_name="Repetitiveness",
            category="craft",
            score=3,
            max_score=5,
            reason="test",
            scope="scenario",
            metric_name="cosine_similarity",
            metric_value=0.35,
        )
        d = rs.to_dict()
        assert d["metric_name"] == "cosine_similarity"
        assert d["metric_value"] == 0.35


# ── RubricGrader ─────────────────────────────────────────────────────────

class TestRubricGrader:
    def test_grade_turn_with_judge(self):
        grades = [{"rubric_id": "test-rubric", "score": 4, "reason": "Good writing"}]
        client = FakeRubricJudgeClient(grades)
        grader = RubricGrader(client=client)

        scenario = _make_scenario()
        state = AccumulatedState(scenario)
        parsed = _make_parsed()
        rubric = _make_rubric()

        scores = grader.grade_turn(
            [rubric], parsed, scenario, scenario.turns[0], state, ["narration 1"]
        )
        assert len(scores) == 1
        assert scores[0].score == 4
        assert scores[0].rubric_id == "test-rubric"
        assert scores[0].action_id == "turn-0"

    def test_grade_turn_no_client_skips_judge(self):
        grader = RubricGrader(client=None)
        scenario = _make_scenario()
        state = AccumulatedState(scenario)
        parsed = _make_parsed()
        rubric = _make_rubric()  # turn-scope, no computed metric

        scores = grader.grade_turn(
            [rubric], parsed, scenario, scenario.turns[0], state, ["n1"]
        )
        # No client → no judge-graded results
        assert scores == []

    def test_grade_turn_computed_only(self):
        grader = RubricGrader(client=None)
        scenario = _make_scenario()
        state = AccumulatedState(scenario)
        parsed = _make_parsed()
        rubric = _make_rubric(computed_metric="cosine_similarity_only")

        scores = grader.grade_turn(
            [rubric], parsed, scenario, scenario.turns[0], state,
            ["dark room with shadows", "dark room with shadows"],
        )
        assert len(scores) == 1
        assert scores[0].metric_name == "cosine_similarity"
        assert scores[0].metric_value is not None
        assert scores[0].score >= 1

    def test_grade_turn_judge_with_cosine_metric(self):
        grades = [{"rubric_id": "rep", "score": 3, "reason": "Moderate variety"}]
        client = FakeRubricJudgeClient(grades)
        grader = RubricGrader(client=client)

        scenario = _make_scenario()
        state = AccumulatedState(scenario)
        parsed = _make_parsed()
        rubric = _make_rubric(rubric_id="rep", computed_metric="cosine_similarity")

        narrations = ["You enter a dark room.", "The dark room echoes."]
        scores = grader.grade_turn(
            [rubric], parsed, scenario, scenario.turns[0], state, narrations,
        )
        assert len(scores) == 1
        assert scores[0].score == 3
        assert scores[0].metric_name == "cosine_similarity"
        assert scores[0].metric_value is not None

    def test_grade_scenario_cosine_only(self):
        grader = RubricGrader(client=None)
        scenario = _make_scenario(n_turns=3)
        state = AccumulatedState(scenario)

        rubric = _make_rubric(
            rubric_id="rep-scenario",
            scope="scenario",
            computed_metric="cosine_similarity_only",
        )

        narrations = [
            "You enter a dark room with stone walls.",
            "A bright garden stretches before you with colorful flowers.",
            "The library is dusty, books line every shelf.",
        ]
        scores = grader.grade_scenario([rubric], narrations, scenario, state)
        assert len(scores) == 1
        assert scores[0].scope == "scenario"
        assert scores[0].metric_name == "cosine_similarity"
        assert scores[0].score >= 1

    def test_grade_scenario_with_judge(self):
        grades = [{"rubric_id": "rep-judge", "score": 5, "reason": "Very varied"}]
        client = FakeRubricJudgeClient(grades)
        grader = RubricGrader(client=client)

        scenario = _make_scenario(n_turns=2)
        state = AccumulatedState(scenario)
        rubric = _make_rubric(
            rubric_id="rep-judge",
            scope="scenario",
            computed_metric="cosine_similarity",
        )

        narrations = ["Narration one.", "Narration two."]
        scores = grader.grade_scenario([rubric], narrations, scenario, state)
        assert len(scores) == 1
        assert scores[0].score == 5
        assert scores[0].metric_name == "cosine_similarity"

    def test_grade_scenario_no_client_skips_judge(self):
        grader = RubricGrader(client=None)
        scenario = _make_scenario(n_turns=2)
        state = AccumulatedState(scenario)
        # scope=scenario, no computed_metric → needs judge
        rubric = _make_rubric(rubric_id="judge-only", scope="scenario")

        scores = grader.grade_scenario([rubric], ["n1", "n2"], scenario, state)
        assert scores == []

    def test_filters_by_scope(self):
        grades = [{"rubric_id": "turn-r", "score": 4, "reason": "ok"}]
        client = FakeRubricJudgeClient(grades)
        grader = RubricGrader(client=client)

        scenario = _make_scenario()
        state = AccumulatedState(scenario)
        parsed = _make_parsed()

        turn_rubric = _make_rubric(rubric_id="turn-r", scope="turn")
        scenario_rubric = _make_rubric(rubric_id="scen-r", scope="scenario")

        # grade_turn should only grade turn-scope rubrics
        scores = grader.grade_turn(
            [turn_rubric, scenario_rubric], parsed, scenario,
            scenario.turns[0], state, ["n1"],
        )
        assert len(scores) == 1
        assert scores[0].rubric_id == "turn-r"

    def test_judge_error_returns_score_0(self):
        class ErrorClient:
            def complete_text(self, *a, **kw):
                raise RuntimeError("API down")

        grader = RubricGrader(client=ErrorClient())
        scenario = _make_scenario()
        state = AccumulatedState(scenario)
        parsed = _make_parsed()
        rubric = _make_rubric()

        scores = grader.grade_turn(
            [rubric], parsed, scenario, scenario.turns[0], state, ["n1"]
        )
        assert len(scores) == 1
        assert scores[0].score == 0
        assert "Judge error" in scores[0].reason

    def test_score_clamping(self):
        """Judge returning out-of-range score should be clamped."""
        grades = [{"rubric_id": "test-rubric", "score": 99, "reason": "oops"}]
        client = FakeRubricJudgeClient(grades)
        grader = RubricGrader(client=client)

        scenario = _make_scenario()
        state = AccumulatedState(scenario)
        parsed = _make_parsed()
        rubric = _make_rubric()

        scores = grader.grade_turn(
            [rubric], parsed, scenario, scenario.turns[0], state, ["n1"]
        )
        assert scores[0].score == 5  # clamped to max

    def test_similarity_to_score_thresholds(self):
        grader = RubricGrader(client=None)
        rubric = _make_rubric()

        assert grader._similarity_to_score(0.10, rubric) == 5
        assert grader._similarity_to_score(0.15, rubric) == 5
        assert grader._similarity_to_score(0.20, rubric) == 4
        assert grader._similarity_to_score(0.30, rubric) == 4
        assert grader._similarity_to_score(0.35, rubric) == 3
        assert grader._similarity_to_score(0.45, rubric) == 3
        assert grader._similarity_to_score(0.50, rubric) == 2
        assert grader._similarity_to_score(0.60, rubric) == 2
        assert grader._similarity_to_score(0.70, rubric) == 1
        assert grader._similarity_to_score(0.90, rubric) == 1


# ── RUBRIC_JUDGE_SYSTEM_PROMPT ───────────────────────────────────────────

class TestJudgePrompt:
    def test_prompt_requests_json(self):
        assert "JSON" in RUBRIC_JUDGE_SYSTEM_PROMPT
        assert "grades" in RUBRIC_JUDGE_SYSTEM_PROMPT

    def test_prompt_mentions_scoring(self):
        assert "5" in RUBRIC_JUDGE_SYSTEM_PROMPT
        assert "1" in RUBRIC_JUDGE_SYSTEM_PROMPT
