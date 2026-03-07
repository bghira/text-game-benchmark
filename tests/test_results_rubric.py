"""Tests for rubric integration in results.py."""

from tgb.checks.base import CheckResult
from tgb.clients.ollama_client import TimingData
from tgb.results import ActionResult, ScenarioResult, BenchmarkRun, print_summary
from tgb.rubric import RubricScore


def _make_rubric_score(rubric_id="voice", score=4, scope="turn", action_id="t-0",
                        metric_name="", metric_value=None):
    return RubricScore(
        rubric_id=rubric_id,
        rubric_name=f"Test {rubric_id}",
        category="craft",
        score=score,
        max_score=5,
        reason="test reason",
        scope=scope,
        action_id=action_id,
        metric_name=metric_name,
        metric_value=metric_value,
    )


class TestActionResultRubrics:
    def test_to_dict_includes_rubric_scores(self):
        ar = ActionResult(
            action_id="t-0",
            action="look",
            checks=[CheckResult("json_valid", True, "ok", "json_structure")],
            rubric_scores=[_make_rubric_score()],
        )
        d = ar.to_dict()
        assert "rubric_scores" in d
        assert len(d["rubric_scores"]) == 1
        assert d["rubric_scores"][0]["rubric_id"] == "voice"
        assert d["rubric_scores"][0]["score"] == 4

    def test_to_dict_no_rubrics_omits_key(self):
        ar = ActionResult(
            action_id="t-0",
            action="look",
            checks=[],
        )
        d = ar.to_dict()
        assert "rubric_scores" not in d


class TestScenarioResultRubrics:
    def test_all_rubric_scores(self):
        ar = ActionResult(
            action_id="t-0",
            action="look",
            rubric_scores=[_make_rubric_score("voice", 4)],
        )
        sr = ScenarioResult(
            scenario="test",
            model="m",
            provider="ollama",
            actions=[ar],
            rubric_scores=[_make_rubric_score("repetitiveness", 3, scope="scenario")],
        )
        all_scores = sr.all_rubric_scores()
        assert len(all_scores) == 2
        ids = [s.rubric_id for s in all_scores]
        assert "voice" in ids
        assert "repetitiveness" in ids

    def test_rubric_summary(self):
        ar = ActionResult(
            action_id="t-0",
            action="look",
            rubric_scores=[
                _make_rubric_score("voice", 4),
                _make_rubric_score("pacing", 3),
            ],
        )
        sr = ScenarioResult(
            scenario="test",
            model="m",
            provider="ollama",
            actions=[ar],
        )
        summary = sr.rubric_summary()
        assert "voice" in summary
        assert summary["voice"]["mean_score"] == 4.0
        assert summary["pacing"]["mean_score"] == 3.0

    def test_rubric_summary_with_metric(self):
        sr = ScenarioResult(
            scenario="test",
            model="m",
            provider="ollama",
            rubric_scores=[_make_rubric_score(
                "repetitiveness", 3, scope="scenario",
                metric_name="cosine_similarity", metric_value=0.35,
            )],
        )
        summary = sr.rubric_summary()
        assert summary["repetitiveness"]["metric_name"] == "cosine_similarity"
        assert summary["repetitiveness"]["metric_value"] == 0.35

    def test_to_dict_includes_rubric_summary(self):
        ar = ActionResult(
            action_id="t-0",
            action="look",
            rubric_scores=[_make_rubric_score("voice", 4)],
        )
        sr = ScenarioResult(
            scenario="test",
            model="m",
            provider="ollama",
            actions=[ar],
        )
        d = sr.to_dict()
        assert "rubrics" in d["summary"]
        assert "voice" in d["summary"]["rubrics"]

    def test_to_dict_no_rubrics_no_key(self):
        sr = ScenarioResult(
            scenario="test",
            model="m",
            provider="ollama",
            actions=[ActionResult(action_id="t-0", action="look")],
        )
        d = sr.to_dict()
        assert "rubrics" not in d["summary"]

    def test_rubric_summary_averages_multiple_turns(self):
        ar1 = ActionResult(
            action_id="t-0",
            action="look",
            rubric_scores=[_make_rubric_score("voice", 5, action_id="t-0")],
        )
        ar2 = ActionResult(
            action_id="t-1",
            action="go",
            rubric_scores=[_make_rubric_score("voice", 3, action_id="t-1")],
        )
        sr = ScenarioResult(
            scenario="test",
            model="m",
            provider="ollama",
            actions=[ar1, ar2],
        )
        summary = sr.rubric_summary()
        assert summary["voice"]["mean_score"] == 4.0  # (5+3)/2
        assert summary["voice"]["n_graded"] == 2


class TestPrintSummaryRubrics:
    def test_print_rubrics(self, capsys):
        ar = ActionResult(
            action_id="t-0",
            action="look",
            checks=[CheckResult("json_valid", True, "ok", "json_structure")],
            rubric_scores=[_make_rubric_score("voice", 4)],
        )
        sr = ScenarioResult(
            scenario="test",
            model="m",
            provider="ollama",
            actions=[ar],
            rubric_scores=[_make_rubric_score(
                "repetitiveness", 3, scope="scenario",
                metric_name="cosine_similarity", metric_value=0.35,
            )],
        )
        run = BenchmarkRun(results=[sr])
        print_summary(run, verbose=True)
        output = capsys.readouterr().out
        assert "Rubric Grades" in output
        assert "voice" in output.lower() or "Voice" in output
        assert "RUBRIC" in output
