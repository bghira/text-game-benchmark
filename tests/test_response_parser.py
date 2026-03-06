"""Tests for response_parser.py."""

import json
import pytest

from tgb.response_parser import (
    extract_json,
    parse_json_lenient,
    clean_response,
    parse_response,
)


class TestExtractJson:
    def test_plain_json(self):
        text = '{"reasoning": "test", "narration": "hello"}'
        assert extract_json(text) == text

    def test_markdown_fenced(self):
        text = '```json\n{"key": "val"}\n```'
        result = extract_json(text)
        assert result == '{"key": "val"}'

    def test_markdown_no_lang(self):
        text = '```\n{"key": "val"}\n```'
        result = extract_json(text)
        assert result == '{"key": "val"}'

    def test_text_around_json(self):
        text = 'Here is my response:\n{"key": "val"}\nDone!'
        result = extract_json(text)
        assert result == '{"key": "val"}'

    def test_no_json(self):
        assert extract_json("no json here") is None

    def test_empty(self):
        assert extract_json("") is None

    def test_just_brace(self):
        assert extract_json("{") is None

    def test_nested_braces(self):
        text = '{"outer": {"inner": 1}}'
        assert extract_json(text) == text


class TestParseJsonLenient:
    def test_valid_json(self):
        result = parse_json_lenient('{"a": 1, "b": "two"}')
        assert result == {"a": 1, "b": "two"}

    def test_python_bools(self):
        result = parse_json_lenient('{"flag": True, "other": False, "val": None}')
        assert result == {"flag": True, "other": False, "val": None}

    def test_concatenated_objects(self):
        text = '{"a": 1}{"b": 2}'
        result = parse_json_lenient(text)
        assert result == {"a": 1, "b": 2}

    def test_invalid_raises(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            parse_json_lenient("not json at all")

    def test_array_returns_empty(self):
        result = parse_json_lenient("[1, 2, 3]")
        assert result == {}


class TestCleanResponse:
    def test_extracts_json(self):
        text = 'Sure!\n{"key": "val"}\n'
        assert clean_response(text) == '{"key": "val"}'

    def test_repairs_truncated(self):
        text = '{"key": "val"'
        result = clean_response(text)
        # Should attempt repair
        assert "{" in result

    def test_empty(self):
        assert clean_response("") == ""

    def test_passthrough(self):
        text = "no json here"
        assert clean_response(text) == "no json here"


class TestParseResponse:
    def test_valid_response(self):
        raw = json.dumps({
            "reasoning": "test",
            "narration": "You see a room.",
            "state_update": {},
            "summary_update": "",
            "xp_awarded": 5,
        })
        result = parse_response(raw)
        assert not result.parse_error
        assert result.parsed_json["narration"] == "You see a room."
        assert not result.is_tool_call

    def test_tool_call(self):
        raw = json.dumps({
            "tool_call": "memory_search",
            "queries": ["white rabbit"],
        })
        result = parse_response(raw)
        assert result.is_tool_call
        assert result.parsed_json["tool_call"] == "memory_search"

    def test_invalid_json(self):
        result = parse_response("This is not JSON at all!!!!")
        assert result.parse_error

    def test_empty(self):
        result = parse_response("")
        assert result.parse_error == "Empty response"

    def test_markdown_wrapped(self):
        inner = {"reasoning": "r", "narration": "n"}
        raw = f"```json\n{json.dumps(inner)}\n```"
        result = parse_response(raw)
        assert not result.parse_error
        assert result.parsed_json["reasoning"] == "r"
