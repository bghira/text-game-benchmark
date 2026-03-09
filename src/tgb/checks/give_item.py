"""Give-item checks — validates give_item against the engine contract.

The engine's give_item contract:
- give_item is a dict with "item" (required str) and one of
  "to_actor_id" or "to_discord_mention" (recipient reference).
- If give_item is used, the model should NOT also use
  player_state_update.inventory_remove for the same item.
"""

from __future__ import annotations

import re
from typing import Any

from tgb.checks.base import CheckResult
from tgb.config import Scenario, TurnSpec
from tgb.prompt_builder import AccumulatedState
from tgb.response_parser import ParsedResponse

# Discord mention format: <@123456> or <@!123456>
DISCORD_MENTION_PATTERN = re.compile(r"^<@!?\d+>$")


def check_give_item_valid(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Validate give_item structure and field types."""
    gi = parsed.parsed_json.get("give_item")
    if gi is None:
        return CheckResult(
            check_id="give_item_valid",
            passed=True,
            detail="No give_item",
            category="give_item",
        )

    if not isinstance(gi, dict):
        return CheckResult(
            check_id="give_item_valid",
            passed=False,
            detail=f"give_item is {type(gi).__name__}, expected dict",
            category="give_item",
        )

    issues: list[str] = []

    # item is required
    item = gi.get("item")
    if not item or not isinstance(item, str) or not item.strip():
        issues.append("'item' is missing or empty")

    # Must have exactly one recipient reference
    to_actor = gi.get("to_actor_id")
    to_mention = gi.get("to_discord_mention")
    has_actor = to_actor is not None
    has_mention = to_mention is not None
    if not has_actor and not has_mention:
        issues.append("need 'to_actor_id' or 'to_discord_mention'")
    elif has_actor and has_mention:
        issues.append("provide exactly one of 'to_actor_id' or 'to_discord_mention', not both")
    if has_actor:
        if isinstance(to_actor, bool):
            issues.append("to_actor_id must not be a bool")
        elif not isinstance(to_actor, (str, int)):
            issues.append(f"to_actor_id is {type(to_actor).__name__}, expected str or int")
    if has_mention:
        if not isinstance(to_mention, str):
            issues.append(f"to_discord_mention is {type(to_mention).__name__}, expected str")
        elif not DISCORD_MENTION_PATTERN.match(to_mention):
            issues.append(f"to_discord_mention '{to_mention}' not a valid mention format")

    if issues:
        return CheckResult(
            check_id="give_item_valid",
            passed=False,
            detail="; ".join(issues),
            category="give_item",
        )
    return CheckResult(
        check_id="give_item_valid",
        passed=True,
        detail=f"give_item valid (item='{item}')",
        category="give_item",
    )


def check_give_item_no_double_remove(
    parsed: ParsedResponse,
    scenario: Scenario,
    turn: TurnSpec,
    state: AccumulatedState,
    params: dict[str, Any],
) -> CheckResult:
    """Check that give_item items aren't also removed via player_state_update.

    The engine handles inventory transfer atomically when give_item is used.
    The model should NOT also remove the item from inventory manually.
    """
    gi = parsed.parsed_json.get("give_item")
    if not isinstance(gi, dict):
        return CheckResult(
            check_id="give_item_no_double_remove",
            passed=True,
            detail="No give_item",
            category="give_item",
        )

    item = gi.get("item", "")
    if not item:
        return CheckResult(
            check_id="give_item_no_double_remove",
            passed=True,
            detail="No item in give_item",
            category="give_item",
        )

    psu = parsed.parsed_json.get("player_state_update")
    if not isinstance(psu, dict):
        return CheckResult(
            check_id="give_item_no_double_remove",
            passed=True,
            detail="No player_state_update",
            category="give_item",
        )

    inv_remove = psu.get("inventory_remove")
    if not isinstance(inv_remove, list):
        return CheckResult(
            check_id="give_item_no_double_remove",
            passed=True,
            detail="No inventory_remove list",
            category="give_item",
        )

    item_lower = item.lower()
    duplicates = [r for r in inv_remove if isinstance(r, str) and r.lower() == item_lower]
    if duplicates:
        return CheckResult(
            check_id="give_item_no_double_remove",
            passed=False,
            detail=f"Item '{item}' in both give_item and inventory_remove — engine handles transfer",
            category="give_item",
        )
    return CheckResult(
        check_id="give_item_no_double_remove",
        passed=True,
        detail="No double-remove for given item",
        category="give_item",
    )
