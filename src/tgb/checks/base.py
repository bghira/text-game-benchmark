"""Base types for checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from tgb.config import CheckSpec, Scenario, TurnSpec
    from tgb.prompt_builder import AccumulatedState
    from tgb.response_parser import ParsedResponse


@dataclass(frozen=True)
class CheckResult:
    """Result of a single check."""

    check_id: str
    passed: bool
    detail: str
    category: str


# Type alias for check functions
CheckFn = Callable[
    ["ParsedResponse", "Scenario", "TurnSpec", "AccumulatedState", dict[str, Any]],
    CheckResult,
]
