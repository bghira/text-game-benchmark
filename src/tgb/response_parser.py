"""JSON extraction from raw model output.

Standalone versions of _extract_json, _parse_json_lenient, _clean_response
ported from ZorkEmulator as pure functions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParsedResponse:
    """Result of parsing a model response."""

    raw: str
    parsed_json: dict[str, Any] = field(default_factory=dict)
    parse_error: str = ""
    is_tool_call: bool = False


def extract_json(text: str) -> str | None:
    """Extract JSON object from raw text, stripping markdown fences."""
    text = text.strip()
    if "```" in text:
        text = re.sub(r"```\w*", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _coerce_python_dict(text: str) -> dict[str, Any] | None:
    """Try interpreting text as a Python dict literal (True/False/None)."""
    try:
        fixed = text.replace("True", "true").replace("False", "false").replace("None", "null")
        result = json.loads(fixed)
        return result if isinstance(result, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def parse_json_lenient(text: str) -> dict[str, Any]:
    """Parse JSON leniently, handling common model output issues."""
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError as exc:
        # Try Python dict coercion
        coerced = _coerce_python_dict(text)
        if coerced is not None:
            return coerced

        if "Extra data" not in str(exc):
            raise

        # Try merging concatenated JSON objects
        merged: dict[str, Any] = {}
        decoder = json.JSONDecoder()
        idx = 0
        length = len(text)
        while idx < length:
            while idx < length and text[idx] in " \t\r\n":
                idx += 1
            if idx >= length:
                break
            try:
                obj, end_idx = decoder.raw_decode(text, idx)
                if isinstance(obj, dict):
                    merged.update(obj)
                idx = end_idx
            except (json.JSONDecodeError, ValueError):
                break
        if merged:
            return merged
        raise


def clean_response(response: str) -> str:
    """Clean raw response text, extracting JSON if present."""
    if not response:
        return response
    cleaned = response.strip()

    json_text = extract_json(cleaned)
    if json_text:
        return json_text

    # Repair truncated object (missing final brace)
    if cleaned.startswith("{") and not cleaned.endswith("}"):
        repaired = f"{cleaned}}}"
        try:
            parsed = parse_json_lenient(repaired)
            if isinstance(parsed, dict) and parsed:
                return repaired
        except Exception:
            pass

    return cleaned


def parse_response(raw_text: str) -> ParsedResponse:
    """Parse a raw model response into a ParsedResponse."""
    if not raw_text or not raw_text.strip():
        return ParsedResponse(raw=raw_text, parse_error="Empty response")

    cleaned = clean_response(raw_text)

    try:
        parsed = parse_json_lenient(cleaned)
    except (json.JSONDecodeError, ValueError) as e:
        return ParsedResponse(raw=raw_text, parse_error=str(e))

    is_tool_call = "tool_call" in parsed

    return ParsedResponse(
        raw=raw_text,
        parsed_json=parsed,
        is_tool_call=is_tool_call,
    )
