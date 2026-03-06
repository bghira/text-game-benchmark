"""Tests for config.py."""

import tempfile
from pathlib import Path

import pytest
import yaml

from tgb.config import (
    load_scenario,
    load_scenarios,
    _build_check_spec,
    _build_turn_spec,
    CheckSpec,
)


def _write_yaml(tmp_path: Path, data: dict, name: str = "test.yaml") -> Path:
    filepath = tmp_path / name
    with open(filepath, "w") as f:
        yaml.dump(data, f)
    return filepath


class TestCheckSpec:
    def test_string_input(self):
        spec = _build_check_spec("json_valid")
        assert spec.check_id == "json_valid"
        assert spec.params == {}

    def test_dict_input(self):
        spec = _build_check_spec({
            "check_id": "xp_range",
            "min": 0,
            "max": 5,
        })
        assert spec.check_id == "xp_range"
        assert spec.params == {"min": 0, "max": 5}

    def test_missing_check_id(self):
        with pytest.raises(ValueError, match="check_id"):
            _build_check_spec({"params": {}})


class TestTurnSpec:
    def test_basic(self):
        turn = _build_turn_spec({"action": "go north"}, 0)
        assert turn.action == "go north"
        assert turn.action_id == "turn-0"

    def test_with_checks(self):
        turn = _build_turn_spec({
            "action": "look",
            "action_id": "look-around",
            "checks": ["json_valid", {"check_id": "xp_range", "max": 5}],
        }, 1)
        assert turn.action_id == "look-around"
        assert len(turn.checks) == 2

    def test_missing_action(self):
        with pytest.raises(ValueError, match="action"):
            _build_turn_spec({}, 0)


class TestLoadScenario:
    def test_minimal(self, tmp_path):
        data = {
            "name": "test-scenario",
            "description": "test",
            "tags": ["test"],
            "tier": "smoke",
            "campaign": {
                "name": "test-campaign",
                "summary": "A test world.",
                "state": {"setting": "test"},
            },
            "player": {
                "user_id": 1,
                "state": {"character_name": "Tester"},
            },
            "turns": [
                {"action": "look around", "checks": ["json_valid"]},
            ],
        }
        filepath = _write_yaml(tmp_path, data)
        scenario = load_scenario(filepath)
        assert scenario.name == "test-scenario"
        assert scenario.campaign.name == "test-campaign"
        assert len(scenario.turns) == 1
        assert scenario.turns[0].checks[0].check_id == "json_valid"

    def test_missing_turns(self, tmp_path):
        data = {
            "campaign": {"name": "test"},
            "turns": [],
        }
        filepath = _write_yaml(tmp_path, data)
        with pytest.raises(ValueError, match="no turns"):
            load_scenario(filepath)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_scenario("/nonexistent/path.yaml")

    def test_default_player(self, tmp_path):
        data = {
            "campaign": {"name": "test", "summary": "test"},
            "turns": [{"action": "go"}],
        }
        filepath = _write_yaml(tmp_path, data)
        scenario = load_scenario(filepath)
        assert scenario.player.user_id == 100000001

    def test_source_material(self, tmp_path):
        data = {
            "campaign": {"name": "test"},
            "turns": [{"action": "go"}],
            "source_material": {
                "documents": [{"key": "doc1", "format": "rulebook", "summary": "rules"}],
                "constraints": ["No magic"],
            },
        }
        filepath = _write_yaml(tmp_path, data)
        scenario = load_scenario(filepath)
        assert scenario.source_material is not None
        assert len(scenario.source_material.documents) == 1
        assert scenario.source_material.constraints == ["No magic"]


class TestLoadScenarios:
    def test_directory(self, tmp_path):
        for name in ["a.yaml", "b.yaml", "_skip.yaml"]:
            data = {
                "name": name.replace(".yaml", ""),
                "campaign": {"name": "test"},
                "turns": [{"action": "go"}],
            }
            _write_yaml(tmp_path, data, name)

        scenarios = load_scenarios([str(tmp_path)])
        assert len(scenarios) == 2  # _skip.yaml excluded
        names = [s.name for s in scenarios]
        assert "_skip" not in names
