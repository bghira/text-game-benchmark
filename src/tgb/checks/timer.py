"""Timer checks — validates timed event usage against the engine contract.

The engine's timer contract:
- set_timer_delay: int 30-300 seconds
- set_timer_event: str describing what happens on expiry
- set_timer_interruptible: bool (default true)
- set_timer_interrupt_action: str|null (context for interruption)
- set_timer_interrupt_scope: "local"|"global" (default "global")

Timers must be grounded in established scene facts, never used for trivial
flavor, and narration must never contain explicit countdowns or seconds.
"""

from __future__ import annotations

import re
from typing import Any

from tgb.checks.base import CheckResult
from tgb.config import Scenario, TurnSpec
from tgb.prompt_builder import AccumulatedState
from tgb.response_parser import ParsedResponse

# Countdown leak patterns — narration must never contain explicit countdowns
COUNTDOWN_PATTERNS = [
    r"\b\d+\s*seconds?\b",
    r"\b\d+\s*minutes?\s+(?:left|remain)",
    r"(?i)\bcountdown\b",
    r"(?i)\btick(?:ing)?\s*(?:clock|down|away)\b",
    r"\u23f0",          # alarm clock emoji
    r"\u23f1",          # stopwatch emoji
    r"\u23f3",          # hourglass emoji
    r"(?i)\btimer\b",   # meta reference to the timer system itself
]


def check_timer_fields_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that timer fields conform to the engine contract.

    Validates:
    - set_timer_delay is int in [30, 300]
    - set_timer_event is a non-empty string
    - set_timer_interruptible is bool if present
    - set_timer_interrupt_scope is "local" or "global" if present
    """
    data = parsed.parsed_json
    delay = data.get("set_timer_delay")

    if delay is None:
        # No timer set — skip unless we expected one
        if params.get("expect_timer", False):
            return CheckResult(
                check_id="timer_fields_valid",
                passed=False,
                detail="Timer expected but no set_timer_delay in response",
                category="timer",
            )
        return CheckResult(
            check_id="timer_fields_valid",
            passed=True,
            detail="No timer set",
            category="timer",
        )

    issues: list[str] = []

    # Delay range
    if not isinstance(delay, (int, float)):
        issues.append(f"set_timer_delay is {type(delay).__name__}, not int")
    else:
        delay_int = int(delay)
        if delay_int < 30 or delay_int > 300:
            issues.append(f"set_timer_delay={delay_int} outside [30, 300]")

    # Event description
    event = data.get("set_timer_event")
    if not event or not isinstance(event, str) or not event.strip():
        issues.append("set_timer_event missing or empty")

    # Interruptible type
    interruptible = data.get("set_timer_interruptible")
    if interruptible is not None and not isinstance(interruptible, bool):
        issues.append(f"set_timer_interruptible is {type(interruptible).__name__}, not bool")

    # Scope
    scope = data.get("set_timer_interrupt_scope")
    if scope is not None and scope not in ("local", "global"):
        issues.append(f"set_timer_interrupt_scope='{scope}' not in ('local', 'global')")

    # Interrupt action type
    interrupt_action = data.get("set_timer_interrupt_action")
    if interrupt_action is not None and not isinstance(interrupt_action, (str, type(None))):
        issues.append(f"set_timer_interrupt_action is {type(interrupt_action).__name__}, not str|null")

    if issues:
        return CheckResult(
            check_id="timer_fields_valid",
            passed=False,
            detail="; ".join(issues),
            category="timer",
        )
    return CheckResult(
        check_id="timer_fields_valid",
        passed=True,
        detail=f"Timer fields valid (delay={delay}s, event={str(event)[:60]})",
        category="timer",
    )


def check_timer_no_countdown_in_narration(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that narration doesn't leak explicit countdowns, seconds, or timer meta-references.

    The system adds its own countdown display automatically. The model must hint
    at urgency narratively (e.g. 'the footsteps grow louder') but never include
    explicit time references like '30 seconds' or countdown emojis.
    """
    narration = parsed.parsed_json.get("narration", "")
    if not isinstance(narration, str) or not narration.strip():
        return CheckResult(
            check_id="timer_no_countdown_in_narration",
            passed=True,
            detail="No narration",
            category="timer",
        )

    # Only check when a timer was actually set
    if "set_timer_delay" not in parsed.parsed_json:
        return CheckResult(
            check_id="timer_no_countdown_in_narration",
            passed=True,
            detail="No timer set — countdown check skipped",
            category="timer",
        )

    leaks: list[str] = []
    for pattern in COUNTDOWN_PATTERNS:
        match = re.search(pattern, narration)
        if match:
            leaks.append(match.group())

    if leaks:
        return CheckResult(
            check_id="timer_no_countdown_in_narration",
            passed=False,
            detail=f"Countdown/time leaked in narration: {leaks[:3]}",
            category="timer",
        )
    return CheckResult(
        check_id="timer_no_countdown_in_narration",
        passed=True,
        detail="No countdown leaks in narration",
        category="timer",
    )


def check_timer_grounded(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that the timer event references established scene elements.

    Timer events must be grounded in known NPCs, hazards, or locations from
    the current state — not spawning unrelated antagonists or random wildlife.

    Params:
        known_elements: list[str] — names/terms that should appear in the timer event
    """
    data = parsed.parsed_json
    event = data.get("set_timer_event")
    if not event or not isinstance(event, str):
        if "set_timer_delay" in data:
            return CheckResult(
                check_id="timer_grounded",
                passed=False,
                detail="Timer set but set_timer_event missing",
                category="timer",
            )
        return CheckResult(
            check_id="timer_grounded",
            passed=True,
            detail="No timer set",
            category="timer",
        )

    known_elements = params.get("known_elements", [])
    if not known_elements:
        # Build from state: character names (full + individual tokens), landmarks, location
        for slug, char_data in state.characters.items():
            if isinstance(char_data, dict):
                name = char_data.get("name", slug)
                known_elements.append(name.lower())
                known_elements.append(slug.lower())
                # Also add individual name tokens (first name, last name)
                for token in name.lower().split():
                    if len(token) >= 3:  # skip short particles
                        known_elements.append(token)
        landmarks = state.campaign_state.get("landmarks", [])
        if isinstance(landmarks, list):
            known_elements.extend(l.lower() for l in landmarks if isinstance(l, str))
        location = state.player_state.get("location", "")
        if location:
            known_elements.append(location.lower())

    if not known_elements:
        # Can't validate grounding without known elements
        return CheckResult(
            check_id="timer_grounded",
            passed=True,
            detail="No known elements to check grounding against",
            category="timer",
        )

    event_lower = event.lower()
    grounded = any(elem in event_lower for elem in known_elements)

    if not grounded:
        return CheckResult(
            check_id="timer_grounded",
            passed=False,
            detail=f"Timer event '{event[:80]}' not grounded in known elements: {known_elements[:5]}",
            category="timer",
        )
    return CheckResult(
        check_id="timer_grounded",
        passed=True,
        detail=f"Timer event references known scene elements",
        category="timer",
    )


def check_timer_delay_appropriate(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that the timer delay matches the urgency of the situation.

    Engine guidance:
    - ~60s for urgent situations
    - ~120s for moderate tension
    - ~180-300s for slow-building tension

    Params:
        expected_urgency: "urgent"|"moderate"|"slow" — what we expect
    """
    data = parsed.parsed_json
    delay = data.get("set_timer_delay")
    if delay is None:
        return CheckResult(
            check_id="timer_delay_appropriate",
            passed=True,
            detail="No timer set",
            category="timer",
        )

    expected_urgency = params.get("expected_urgency", "")
    if not expected_urgency:
        return CheckResult(
            check_id="timer_delay_appropriate",
            passed=True,
            detail=f"Timer delay={delay}s (no urgency expectation specified)",
            category="timer",
        )

    delay_int = int(delay)
    ranges = {
        "urgent": (30, 90),
        "moderate": (90, 180),
        "slow": (150, 300),
    }

    expected_range = ranges.get(expected_urgency, (30, 300))
    lo, hi = expected_range
    if delay_int < lo or delay_int > hi:
        return CheckResult(
            check_id="timer_delay_appropriate",
            passed=False,
            detail=f"Timer delay={delay_int}s doesn't match '{expected_urgency}' urgency (expected {lo}-{hi}s)",
            category="timer",
        )
    return CheckResult(
        check_id="timer_delay_appropriate",
        passed=True,
        detail=f"Timer delay={delay_int}s appropriate for '{expected_urgency}' urgency",
        category="timer",
    )


def check_no_gratuitous_timer(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that timers aren't set for trivial actions that don't warrant urgency.

    If the scenario flags a turn as non-urgent (via params), a timer
    should NOT be set. Timers should force decisions or advance the plot,
    never be used for flavor.

    Params:
        expect_no_timer: bool — if True, setting a timer here is a failure
    """
    expect_no_timer = params.get("expect_no_timer", False)
    if not expect_no_timer:
        return CheckResult(
            check_id="no_gratuitous_timer",
            passed=True,
            detail="No restriction on timer usage for this turn",
            category="timer",
        )

    has_timer = "set_timer_delay" in parsed.parsed_json
    if has_timer:
        event = parsed.parsed_json.get("set_timer_event", "")
        return CheckResult(
            check_id="no_gratuitous_timer",
            passed=False,
            detail=f"Timer set on a non-urgent turn: '{str(event)[:80]}'",
            category="timer",
        )
    return CheckResult(
        check_id="no_gratuitous_timer",
        passed=True,
        detail="No timer set (correct for non-urgent turn)",
        category="timer",
    )
